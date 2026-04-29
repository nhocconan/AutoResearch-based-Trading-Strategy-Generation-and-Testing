#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation (>1.5x 20-period average)
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades in bear markets
# Volume confirmation filters weak signals, reducing false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_WilliamsR_MeanRev_1dEMA34_VolumeSpike"
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
    
    # Calculate Williams %R on 12h timeframe (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate 20-period average volume for confirmation (on 12h timeframe)
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
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_williams_r = williams_r[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR price closes below 1d EMA34
            if curr_williams_r > -20 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR price closes above 1d EMA34
            if curr_williams_r < -80 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 from below (oversold bounce) + price above 1d EMA34 + volume confirmation
            if (curr_williams_r > -80 and 
                curr_close > curr_ema_1d and 
                vol_confirm and
                i > start_idx and williams_r[i-1] <= -80):  # Ensure crossover from below
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 from above (overbought rejection) + price below 1d EMA34 + volume confirmation
            elif (curr_williams_r < -20 and 
                  curr_close < curr_ema_1d and 
                  vol_confirm and
                  i > start_idx and williams_r[i-1] >= -20):  # Ensure crossover from above
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals