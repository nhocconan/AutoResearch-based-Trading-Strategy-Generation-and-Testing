#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use prior day's OHLC to calculate weekly pivot points
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Weekly pivot point: (H + L + C) / 3
    pp = (prev_high + prev_low + prev_close) / 3
    # Weekly resistance and support levels
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    r3 = high_1d + 2 * (pp - prev_low)
    s3 = low_1d - 2 * (prev_high - pp)
    
    # Align pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.8x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # ATR(14) for stop loss calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume and ATR calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(avg_vol[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation
            if price > r3_aligned[i] and vol > 1.8 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S3 with volume confirmation
            elif price < s3_aligned[i] and vol > 1.8 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1 or hits ATR-based stop
            if price < s1_aligned[i] or price < (signals[i-1] * position_size * 0 + (entry_price if 'entry_price' in locals() else 0) - 2.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                # Track entry price for stop loss
                if i == start or signals[i-1] == 0:
                    entry_price = price
        elif position == -1:
            # Exit short: price breaks above R1 or hits ATR-based stop
            if price > r1_aligned[i] or price > (signals[i-1] * position_size * 0 + (entry_price if 'entry_price' in locals() else 0) + 2.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
                # Track entry price for stop loss
                if i == start or signals[i-1] == 0:
                    entry_price = price
    
    return signals

name = "1d_1w_Pivot_Breakout_Weekly_ATR"
timeframe = "1d"
leverage = 1.0