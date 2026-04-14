#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly pivot-based breakout on daily chart with volume confirmation and ADX trend filter
# Long when price breaks above weekly R3 pivot AND ADX > 20 (trending) AND volume > 1.5x 20-period average
# Short when price breaks below weekly S3 pivot AND ADX > 20 AND volume > 1.5x 20-period average
# Exit when price crosses back to opposite weekly pivot level (S3 for longs, R3 for shorts)
# Weekly pivots provide strong support/resistance levels; ADX filters for trending conditions; volume confirms breakout strength
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while capturing significant moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using weekly OHLC)
    # Pivot = (H + L + C) / 3
    # R3 = H + 2*(Pivot - L)
    # S3 = L - 2*(H - Pivot)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r3 = weekly_high + 2.0 * (pivot - weekly_low)
    s3 = weekly_low - 2.0 * (weekly_high - pivot)
    
    # Align weekly pivot levels to daily timeframe (wait for weekly bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate ADX for trend strength (14-period)
    # ADX requires +DI and -DI calculation
    high_diff = np.diff(high, prepend=high[0])
    low_diff = np.diff(low, prepend=low[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First period
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth the values (14-period)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    plus_di = np.where(tr_sum > 0, 100.0 * plus_dm_sum / tr_sum, 0.0)
    minus_di = np.where(tr_sum > 0, 100.0 * minus_dm_sum / tr_sum, 0.0)
    
    dx = np.where((plus_di + minus_di) > 0, 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above weekly R3 + ADX > 20 + volume confirmation
            if (price > r3_aligned[i] and adx[i] > 20.0 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below weekly S3 + ADX > 20 + volume confirmation
            elif (price < s3_aligned[i] and adx[i] > 20.0 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below weekly S3 (opposite support)
            if price < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above weekly R3 (opposite resistance)
            if price > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyPivot_R3S3_ADX_Volume"
timeframe = "1d"
leverage = 1.0