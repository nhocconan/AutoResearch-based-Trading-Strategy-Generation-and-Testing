#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h EMA50 trend filter and volume confirmation (>2.0x 24-period average)
# Williams %R identifies overbought/oversold conditions; extreme readings (>80 or <20) signal potential reversals
# 12h EMA50 ensures alignment with intermediate trend to avoid counter-trend trades in ranging markets
# Volume spike (>2.0x average) confirms conviction behind the reversal move
# Discrete sizing (0.25) and tight conditions target 50-150 trades over 4 years on 6h timeframe
# Works in bull/bear: mean reversion effective in range, trend filter prevents whipsaws in strong trends

name = "6h_WilliamsR_MeanRev_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R on 6h timeframe (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 24-period average volume for confirmation (on 6h timeframe)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 24)  # 12h EMA50, Williams %R, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_vol_ma = vol_ma_24[i]
        curr_volume = volume[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 24-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Williams %R rises above -20 (overbought) OR price closes below 12h EMA50
            if curr_williams_r > -20 or curr_close < curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -80 (oversold) OR price closes above 12h EMA50
            if curr_williams_r < -80 or curr_close > curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R below -80 (oversold) + price above 12h EMA50 + volume confirmation
            if (curr_williams_r < -80 and 
                curr_close > curr_ema_12h and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R above -20 (overbought) + price below 12h EMA50 + volume confirmation
            elif (curr_williams_r > -20 and 
                  curr_close < curr_ema_12h and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals