#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# Long when price breaks above R3 AND 1w EMA50 uptrend AND volume > 2x 24-period median.
# Short when price breaks below S3 AND 1w EMA50 downtrend AND volume > 2x 24-period median.
# Camarilla levels provide high-probability reversal/breakout points; 1w EMA50 filters for primary trend; volume spike confirms institutional interest.
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years).

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_range = (high_1d - low_1d) * 1.1 / 2
    r3 = close_1d + camarilla_range
    s3 = close_1d - camarilla_range
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 24-period volume median for volume confirmation (24 * 12h = 12d lookback)
    vol_median_24 = pd.Series(volume).rolling(window=24, min_periods=24).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_median_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1w EMA50 direction
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 2x 24-period volume median
        if vol_median_24[i] <= 0 or np.isnan(vol_median_24[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_24[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 AND uptrend AND volume spike
            if curr_high > r3_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND downtrend AND volume spike
            elif curr_low < s3_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 OR trend turns down
            if curr_low < s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 OR trend turns up
            if curr_high > r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals