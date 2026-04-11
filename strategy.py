#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and volume confirmation
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long: Bull Power > 0 AND Bear Power < 0 (bulls in control) AND price > EMA(50) on 12h (uptrend) AND volume > 1.5x 20-period avg
# - Short: Bear Power > 0 AND Bull Power < 0 (bears in control) AND price < EMA(50) on 12h (downtrend) AND volume > 1.5x 20-period avg
# - Exit: Power values cross zero or price returns to EMA(13)
# - Uses discrete sizing 0.25 to limit drawdown
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Works in both bull and bear by measuring actual bull/bear power behind moves

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
    
    # Load 12h data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-compute 6h EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = ema_13 - low   # Bear Power = EMA(13) - Low
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Elder Ray values
        current_bull_power = bull_power[i]
        current_bear_power = bear_power[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 12h EMA trend bias
        ema_bias_long = close_price > ema_50_12h_aligned[i]
        ema_bias_short = close_price < ema_50_12h_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power > 0 (bulls in control) AND Bear Power < 0 (no bear resistance)
        #       AND price above 12h EMA(50) (uptrend) AND volume confirmation
        if (current_bull_power > 0 and current_bear_power < 0 and 
            ema_bias_long and vol_confirm):
            enter_long = True
        
        # Short: Bear Power > 0 (bears in control) AND Bull Power < 0 (no bull resistance)
        #        AND price below 12h EMA(50) (downtrend) AND volume confirmation
        if (current_bear_power > 0 and current_bull_power < 0 and 
            ema_bias_short and vol_confirm):
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bull Power turns negative OR price returns to EMA(13)
            exit_long = (current_bull_power <= 0) or (abs(close_price - ema_13[i]) < 0.001 * close_price)
        elif position == -1:
            # Exit short if Bear Power turns negative OR price returns to EMA(13)
            exit_short = (current_bear_power <= 0) or (abs(close_price - ema_13[i]) < 0.001 * close_price)
        
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