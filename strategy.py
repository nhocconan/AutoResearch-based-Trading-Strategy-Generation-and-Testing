#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 fade with 1d trend filter and volume spike confirmation
# Fade at R3/S3 levels when price shows rejection (wick > 50% of body) AND volume spike
# Continuation breakout at R4/S4 levels with volume spike AND 1d EMA50 alignment
# Uses Camarilla pivot structure for mean reversion in ranges and breakout in trends
# Effective in both bull (continuation longs) and bear (continuation shorts) markets
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_R3S3_Fade_R4S4_Breakout_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Volume confirmation: spike > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    # Typical Price = (H + L + C) / 3
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    range_1d = df_1d['high'].values - df_1d['low'].values
    
    # Camarilla levels: R3 = TP + 1.1 * range/2, S3 = TP - 1.1 * range/2
    # R4 = TP + 1.1 * range, S4 = TP - 1.1 * range
    camarilla_tp = typical_price_1d
    camarilla_range = range_1d
    
    r3 = camarilla_tp + 1.1 * camarilla_range / 2.0
    s3 = camarilla_tp - 1.1 * camarilla_range / 2.0
    r4 = camarilla_tp + 1.1 * camarilla_range
    s4 = camarilla_tp - 1.1 * camarilla_range
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate rejection condition: wick > 50% of body
        body_size = abs(close[i] - open_price[i])
        upper_wick = high[i] - max(close[i], open_price[i])
        lower_wick = min(close[i], open_price[i]) - low[i]
        max_wick = max(upper_wick, lower_wick)
        
        # Avoid division by zero
        if body_size == 0:
            rejection = False
        else:
            rejection = max_wick > (0.5 * body_size)
        
        if position == 0:
            # Long fade at S3: price rejects S3 with volume spike AND above 1d EMA50 (bullish bias)
            if (low[i] <= s3_aligned[i] and 
                rejection and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short fade at R3: price rejects R3 with volume spike AND below 1d EMA50 (bearish bias)
            elif (high[i] >= r3_aligned[i] and 
                  rejection and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            # Long breakout at R4: price breaks R4 with volume spike AND above 1d EMA50
            elif (high[i] > r4_aligned[i] and 
                  volume_spike[i] and 
                  close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout at S4: price breaks S4 with volume spike AND below 1d EMA50
            elif (low[i] < s4_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches R3 (fade level) or loses volume/spike
            if (high[i] >= r3_aligned[i] and rejection) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches S3 (fade level) or loses volume/spike
            if (low[i] <= s3_aligned[i] and rejection) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals