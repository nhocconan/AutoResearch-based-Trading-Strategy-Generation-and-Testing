# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot breakouts with trend filter (1d EMA) and volume confirmation
# work in both bull and bear markets by capturing institutional reversal levels with momentum.
# Timeframe 12h reduces trade frequency to avoid fee drag while maintaining responsiveness.
# Volume filter ensures breakouts have institutional participation.
# Camarilla levels (R3/S3) act as strong support/resistance where price often reverses or accelerates.
# Trend filter (1d EMA34) ensures trades align with higher-timeframe direction.
# Expected trades: 20-40 per year, within optimal range for 12h timeframe.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of previous day)
    # Camarilla: R4 = C + ((H-L) * 1.5), R3 = C + ((H-L) * 1.25)
    #          S3 = C - ((H-L) * 1.25), S4 = C - ((H-L) * 1.5)
    # where C = (H+L+C)/3 (typical price)
    # We use previous day's data to avoid look-ahead
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_prev = typical_price.shift(1)  # Previous day's typical price
    high_prev = df_1d['high'].shift(1)
    low_prev = df_1d['low'].shift(1)
    
    # Calculate Camarilla levels using previous day's range
    range_prev = high_prev - low_prev
    camarilla_r3 = typical_price_prev + (range_prev * 1.25)
    camarilla_s3 = typical_price_prev - (range_prev * 1.25)
    
    # Align Camarilla levels to 12h timeframe (they update daily)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Volume filter: current 12h volume > 1.8 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: price breaks above R3 + above 1d EMA trend + volume filter
            if close[i] > r3_level and close[i] > ema34_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + below 1d EMA trend + volume filter
            elif close[i] < s3_level and close[i] < ema34_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 (reversal signal) or trend fails
            if close[i] < s3_level or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 (reversal signal) or trend fails
            if close[i] > r3_level or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals