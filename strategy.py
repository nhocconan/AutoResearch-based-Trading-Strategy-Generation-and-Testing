#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume spike (>2x 20-period average)
# Williams %R identifies overbought/oversold conditions; extreme readings (< -80 or > -20) signal potential reversals.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws in both bull/bear markets.
# Volume spike filter (>2x average) confirms significant market interest, reducing false signals.
# Discrete position sizing (0.25) minimizes fee churn while maintaining meaningful exposure.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 12h timeframe (period=14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 20-period average volume for spike confirmation (on 12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14, 20)  # 1d EMA34, Williams %R, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 2x 20-period average
        vol_spike = curr_volume > 2.0 * curr_vol_ma
        
        # Williams %R conditions
        oversold = curr_wr < -80  # Oversold condition
        overbought = curr_wr > -20  # Overbought condition
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Williams %R returns from oversold OR trend turns bearish
            if (curr_wr > -50 or curr_close < curr_ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns from overbought OR trend turns bullish
            if (curr_wr < -50 or curr_close > curr_ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R oversold AND above 1d EMA34 AND volume spike
            if (oversold and 
                curr_close > curr_ema_1d and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought AND below 1d EMA34 AND volume spike
            elif (overbought and 
                  curr_close < curr_ema_1d and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals