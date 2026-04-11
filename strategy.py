#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d/1w regime filter
# - Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13)
# - Long when Bull Power > 0 and rising (bullish momentum) AND 1d EMA50 > 1d EMA200 (bullish regime)
# - Short when Bear Power < 0 and falling (bearish momentum) AND 1d EMA50 < 1d EMA200 (bearish regime)
# - Uses 1w trend filter: only trade long if price > 1w EMA50, short if price < 1w EMA50
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h
# - Works in bull markets (trend following with momentum) and bear markets (counter-trend reversals at extremes)

name = "6h_1d_1w_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return signals
    
    # Pre-compute 1d EMAs for regime filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMAs to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Calculate 2-period change for momentum
    bull_power_change = bull_power - np.roll(bull_power, 2)
    bear_power_change = bear_power - np.roll(bear_power, 2)
    # Handle first two values
    bull_power_change[:2] = 0
    bear_power_change[:2] = 0
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        bullish_regime = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        bearish_regime = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        # Trend filter
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # Elder Ray signals with momentum
        bullish_momentum = bull_power[i] > 0 and bull_power_change[i] > 0
        bearish_momentum = bear_power[i] < 0 and bear_power_change[i] < 0
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bullish momentum + bullish regime + price above weekly EMA
        if bullish_momentum and bullish_regime and price_above_1w_ema:
            enter_long = True
        
        # Short: Bearish momentum + bearish regime + price below weekly EMA
        if bearish_momentum and bearish_regime and price_below_1w_ema:
            enter_short = True
        
        # Exit conditions: opposite momentum or regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bearish momentum OR regime turns bearish
            exit_long = bearish_momentum or (not bullish_regime)
        elif position == -1:
            # Exit short if bullish momentum OR regime turns bullish
            exit_short = bullish_momentum or (not bearish_regime)
        
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