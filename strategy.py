#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 12h trend filter and volume confirmation
# - Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Long: Bull Power > 0 AND Bear Power rising (improving) AND price > 12h EMA50 AND volume > 1.5x 20-period avg
# - Short: Bear Power < 0 AND Bull Power falling (weakening) AND price < 12h EMA50 AND volume > 1.5x 20-period avg
# - Exit: Opposite signal appears or ATR-based stop (2.5 ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Elder Ray measures bull/bear strength relative to trend (EMA)
# - 12h EMA50 filter ensures we trade with higher timeframe trend
# - Volume confirmation avoids low-participation false signals

name = "6h_12h_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute Elder Ray components (6h timeframe)
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Pre-compute previous values for crossover/change detection
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = bull_power[0]
    bear_power_prev[0] = bear_power[0]
    
    # Pre-compute volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR for stoploss (6h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(bull_power_prev[i]) or
            np.isnan(bear_power_prev[i]) or np.isnan(volume_sma_20[i]) or np.isnan(atr_20[i]) or
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Elder Ray values
        bull_current = bull_power[i]
        bear_current = bear_power[i]
        bull_previous = bull_power_prev[i]
        bear_previous = bear_power_prev[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Trend filter: price vs 12h EMA50
        price_above_12h_ema = close_price > ema50_12h_aligned[i]
        price_below_12h_ema = close_price < ema50_12h_aligned[i]
        
        # Elder Ray momentum: improving/declining
        bull_rising = bull_current > bull_previous
        bull_falling = bull_current < bull_previous
        bear_rising = bear_current > bear_previous  # less negative = improving
        bear_falling = bear_current < bear_previous  # more negative = worsening
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power positive AND rising AND price above 12h EMA50 AND volume confirmation
        if bull_current > 0 and bull_rising and price_above_12h_ema and vol_confirm:
            enter_long = True
        
        # Short: Bear Power negative AND falling (more negative) AND price below 12h EMA50 AND volume confirmation
        if bear_current < 0 and bear_falling and price_below_12h_ema and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bear Power turns positive (bulls losing) OR ATR stop
            exit_long = (bear_current >= 0) or (close_price <= entry_price - 2.5 * atr_20[i])
        elif position == -1:
            # Exit short if Bull Power turns negative (bears losing) OR ATR stop
            exit_short = (bull_current <= 0) or (close_price >= entry_price + 2.5 * atr_20[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals