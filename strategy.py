#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Breakout with Volume and 12h Trend Filter
# Uses daily Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout)
# 12h EMA (50) provides trend direction to filter breakouts
# Volume confirmation (>1.5x average) ensures momentum
# Designed to work in both bull and bear markets by trading breakouts in direction of higher timeframe trend
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivots and 12h EMA ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = Close + Range * 1.1/2
    # R3 = Close + Range * 1.1/4
    # S3 = Close - Range * 1.1/4
    # S4 = Close - Range * 1.1/2
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    pivot = typical_price.values
    r4 = (df_1d['close'] + range_hl * 1.1 / 2).values
    r3 = (df_1d['close'] + range_hl * 1.1 / 4).values
    s3 = (df_1d['close'] - range_hl * 1.1 / 4).values
    s4 = (df_1d['close'] - range_hl * 1.1 / 2).values
    
    # Align Camarilla levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Load 12h EMA data
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5x average volume (30-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=30, min_periods=30).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 12h EMA calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 12h EMA
        trend_up = price > ema_50_12h_aligned[i]
        trend_down = price < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume filter and uptrend
            if price > r4_aligned[i] and vol > 1.5 * avg_vol[i] and trend_up:
                position = 1
                signals[i] = position_size
            # Short breakdown: price breaks below S4 with volume filter and downtrend
            elif price < s4_aligned[i] and vol > 1.5 * avg_vol[i] and trend_down:
                position = -1
                signals[i] = -position_size
            # Long fade: price rejects at S3 with volume filter and uptrend
            elif price < s3_aligned[i] and vol > 1.5 * avg_vol[i] and trend_up and i > start and close[i-1] >= s3_aligned[i-1]:
                position = 1
                signals[i] = position_size
            # Short fade: price rejects at R3 with volume filter and downtrend
            elif price > r3_aligned[i] and vol > 1.5 * avg_vol[i] and trend_down and i > start and close[i-1] <= r3_aligned[i-1]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below pivot (mean reversion) or stops at R3
            if price < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above pivot (mean reversion) or stops at S3
            if price > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Camarilla_Pivot_Breakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0