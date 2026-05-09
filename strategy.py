#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_PivotZone_1wTrend_1dVol"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Get daily data for pivot levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Weekly trend: EMA50
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily pivot points (using prior day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current 6h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(100, 50)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1w_aligned[i]
        pivot_val = pivot_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: price above S3 + above weekly EMA50 + volume filter
            if close[i] > s3_val and close[i] > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price below R3 + below weekly EMA50 + volume filter
            elif close[i] < r3_val and close[i] < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below pivot
            if close[i] < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above pivot
            if close[i] > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

if __name__ == "__main__":
    # Quick self-test
    import yfinance as yf
    data = yf.download("BTC-USD", start="2021-01-01", end="2024-12-31", interval="60m")
    data.reset_index(inplace=True)
    data.rename(columns={'Datetime': 'open_time'}, inplace=True)
    data['open_time'] = pd.to_datetime(data['open_time'])
    sig = generate_signals(data)
    print(f"Signals generated: {len(sig)}")
    print(f"Non-zero signals: {np.sum(sig != 0)}")