#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long: Bull Power > 0 AND Bear Power < 0 (bulls in control) AND 1d EMA(50) uptrend AND volume > 1.3x 20-period avg
# - Short: Bear Power > 0 AND Bull Power < 0 (bears in control) AND 1d EMA(50) downtrend AND volume > 1.3x 20-period avg
# - Exit: Power values converge toward zero (loss of momentum)
# - Uses 6h timeframe for lower frequency (12-37 trades/year target) to minimize fee drag
# - Elder Ray captures momentum shifts; 1d EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation reduces false signals in low-participation moves

name = "6h_1d_elder_ray_volume_v2"
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
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = ema_13 - low   # Bear Power = EMA - Low
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Elder Ray values
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power positive, Bear Power negative (bulls in control), volume confirmation, long bias
        if bull_val > 0 and bear_val < 0 and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short: Bear Power positive, Bull Power negative (bears in control), volume confirmation, short bias
        if bear_val > 0 and bull_val < 0 and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions: Power values converge toward zero (loss of momentum)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when Bull Power weakens (< 0) or Bear Power strengthens (> 0)
            exit_long = bull_val <= 0 or bear_val >= 0
        elif position == -1:
            # Exit short when Bear Power weakens (< 0) or Bull Power strengthens (> 0)
            exit_short = bear_val <= 0 or bull_val >= 0
        
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