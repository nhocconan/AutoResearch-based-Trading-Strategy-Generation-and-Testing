#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with Daily Volume Spike and Daily Trend Filter
# Takes long when price breaks above 12h Camarilla resistance (R3/R4) with volume spike and daily EMA > EMA(50)
# Takes short when price breaks below 12h Camarilla support (S3/S4) with volume spike and daily EMA < EMA(50)
# Exits when price crosses back below/above the 12h Camarilla pivot (midpoint)
# Camarilla levels are calculated from daily OHLC, providing institutional support/resistance levels
# Works in bull/bear markets by following institutional levels with volume confirmation
# Target: 15-30 trades per symbol over 4 years (4-7.5/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and daily data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h Camarilla levels from daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculation: based on previous day's range
    range_1d = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_r3 = camarilla_pivot + (range_1d * 1.1 / 2)
    camarilla_r4 = camarilla_pivot + (range_1d * 1.1)
    camarilla_s3 = camarilla_pivot - (range_1d * 1.1 / 2)
    camarilla_s4 = camarilla_pivot - (range_1d * 1.1)
    
    # Calculate daily EMA for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for Camarilla and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_12h_current = volume[i]  # Current 12h volume
        
        if position == 0:
            # Long setup: break above Camarilla R3/R4 with volume spike and bullish trend
            if ((price > camarilla_r3_aligned[i] or price > camarilla_r4_aligned[i]) and 
                vol_12h_current > 1.8 * vol_ma_1d_aligned[i] and  # Volume spike
                close[i] > ema_1d_aligned[i]):                  # Bullish trend (price above daily EMA)
                position = 1
                signals[i] = position_size
            # Short setup: break below Camarilla S3/S4 with volume spike and bearish trend
            elif ((price < camarilla_s3_aligned[i] or price < camarilla_s4_aligned[i]) and 
                  vol_12h_current > 1.8 * vol_ma_1d_aligned[i] and  # Volume spike
                  close[i] < ema_1d_aligned[i]):                  # Bearish trend (price below daily EMA)
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Camarilla pivot
            if price < camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Camarilla pivot
            if price > camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Breakout_1dVolume_EMA"
timeframe = "12h"
leverage = 1.0