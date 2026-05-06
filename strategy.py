#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 1d structure for Camarilla pivot levels (key support/resistance), 4h EMA50 for trend alignment (reduces whipsaw)
# Volume spike (>1.5x 20-bar average) confirms breakout strength
# Discrete sizing 0.20 to limit fee drag; target 60-150 total trades over 4 years (15-37/year)
# Session filter: 08-20 UTC to avoid low-liquidity periods
# Proven pattern: price channel breakouts with volume confirmation work on BTC/ETH in both bull/bear markets

name = "1h_Camarilla_R3S3_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_1d) < 50 or len(df_4h) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 1d EMA34 trend filter (for additional confirmation)
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 4h EMA50 trend filter
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume spike filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Calculate 1d Camarilla pivot levels (using previous 1d bar)
    # Camarilla: R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    camarilla_high = []
    camarilla_low = []
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_high.append(np.nan)
            camarilla_low.append(np.nan)
        else:
            h = high_1d[i-1]
            l = low_1d[i-1]
            c = close_1d[i-1]
            r3 = c + ((h - l) * 1.1 / 4)
            s3 = c - ((h - l) * 1.1 / 4)
            camarilla_high.append(r3)
            camarilla_low.append(s3)
    
    camarilla_high = np.array(camarilla_high)
    camarilla_low = np.array(camarilla_low)
    
    # Align HTF indicators to 1h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    volume_filter_aligned = align_htf_to_ltf(prices, df_4h, volume_filter)  # Align volume to 4h to avoid noise
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend (price > EMA50_4h) AND volume spike
            if close[i] > camarilla_high_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short breakdown: price < S3 AND downtrend (price < EMA50_4h) AND volume spike
            elif close[i] < camarilla_low_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price retests S3 from above (trend reversal)
            if close[i] <= camarilla_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retests R3 from below (trend reversal)
            if close[i] >= camarilla_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals