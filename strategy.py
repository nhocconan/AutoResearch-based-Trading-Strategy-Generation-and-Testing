#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 Breakout with 1w EMA34 Trend Filter and Volume Spike
# Long when price breaks above R3 with volume > 1.5x 20-period average and close > 1w EMA34
# Short when price breaks below S3 with volume > 1.5x 20-period average and close < 1w EMA34
# Exit when price re-enters the Camarilla H-L range (between H3 and L3)
# Uses discrete position sizing (0.30) to balance capture and risk.
# Camarilla pivots provide intraday support/resistance levels, 1w EMA34 filters for higher-timeframe trend.
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to avoid overtrading.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1w trend.

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H = high, L = low, C = close of previous day
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    # Camarilla levels: H3, L3, H4, L4 (we use H3 and L3 for breakout)
    # H3 = C + (H-L)*1.1/4
    # L3 = C - (H-L)*1.1/4
    H3 = C + (H - L) * 1.1 / 4
    L3 = C - (H - L) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (previous day's levels are known at 12h open)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # 1w EMA34 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_H3 = H3_aligned[i]
        curr_L3 = L3_aligned[i]
        curr_close = close[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price re-enters Camarilla H-L range (close < H3)
            if curr_close < curr_H3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price re-enters Camarilla H-L range (close > L3)
            if curr_close > curr_L3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation
            vol_ok = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above R3 with volume confirmation and close > 1w EMA34
            if curr_close > curr_H3 and vol_ok and curr_close > curr_ema34_1w:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below S3 with volume confirmation and close < 1w EMA34
            elif curr_close < curr_L3 and vol_ok and curr_close < curr_ema34_1w:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals