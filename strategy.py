# 4H_CAMARILLA_R3_S3_BREAKOUT_1D_EMA34_VOLUME_SPIKE_TREND
# Strategy uses Camarilla R3/S3 levels from daily pivots with volume spike and EMA34 trend filter
# Works in bull/bear: Long at R3 breakout in uptrend, short at S3 breakdown in downtrend
# Target: 20-35 trades/year per symbol (80-140 total over 4 years)
# Volume confirmation reduces false breakouts, EMA34 filter avoids counter-trend trades

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each day
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_r3[i] = close_1d[i]
            camarilla_s3[i] = close_1d[i]
        else:
            # Use previous day's OHLC for today's levels
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            camarilla_r3[i] = pc + (ph - pl) * 1.1 / 2
            camarilla_s3[i] = pc - (ph - pl) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_1d_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_1d_period + 1)) + 
                         ema_1d[i-1] * (1 - (2 / (ema_1d_period + 1))))
    
    # Align EMA34 to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate volume spike (current volume vs 20-period average)
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    if n >= vol_ma_period:
        for i in range(vol_ma_period - 1, n):
            vol_ma[i] = np.mean(volume[i - vol_ma_period + 1:i + 1])
    
    volume_spike = np.full(n, False)
    for i in range(vol_ma_period - 1, n):
        if vol_ma[i] > 0:
            volume_spike[i] = volume[i] > (vol_ma[i] * 1.5)  # 50% above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla levels, EMA34, and volume MA
    start_idx = max(34, vol_ma_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        ema_trend = ema_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and uptrend (price > EMA34)
            if (price > r3_level and vol_spike and price > ema_trend):
                signals[i] = size
                position = 1
            # Short: Price breaks below S3 with volume spike and downtrend (price < EMA34)
            elif (price < s3_level and vol_spike and price < ema_trend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price returns below R3 or trend fails
            if price < r3_level or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price returns above S3 or trend fails
            if price > s3_level or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_Camarilla_R3_S3_Breakout_1D_EMA34_Volume_Spike_Trend"
timeframe = "4h"
leverage = 1.0