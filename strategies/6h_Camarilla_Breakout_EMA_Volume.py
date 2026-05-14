#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with daily trend filter and volume confirmation
# Camarilla levels provide institutional support/resistance. Breakouts with volume and daily EMA trend
# filter capture institutional flow while avoiding false breakouts in chop.
# Works in bull/bear by using daily EMA trend filter (long only above EMA, short only below EMA)
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate previous day's Camarilla levels
    # Based on previous day's range: H-L
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day will have NaN due to roll, handle with fill
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    range_1d = prev_high - prev_low
    
    # Camarilla levels: Close ± (range * multiplier)
    # Key levels for breakout: R3, R4, S3, S4
    camarilla_r3 = prev_close + range_1d * 1.1 / 4
    camarilla_r4 = prev_close + range_1d * 1.1 / 2
    camarilla_s3 = prev_close - range_1d * 1.1 / 4
    camarilla_s4 = prev_close - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (no delay needed as they're based on closed daily candle)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 1.5x average volume (24-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 24  # for volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above R4 with volume filter AND above daily EMA50
            if (price > r4_aligned[i] and price > ema_50_1d_aligned[i] and 
                vol > 1.5 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below S4 with volume filter AND below daily EMA50
            elif (price < s4_aligned[i] and price < ema_50_1d_aligned[i] and 
                  vol > 1.5 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below R3 OR below daily EMA50
            if price < r3_aligned[i] or price < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above S3 OR above daily EMA50
            if price > s3_aligned[i] or price > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Camarilla_Breakout_EMA_Volume"
timeframe = "6h"
leverage = 1.0