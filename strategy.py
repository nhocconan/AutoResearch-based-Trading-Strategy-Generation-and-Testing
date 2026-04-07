#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 1-week pivot direction and 1-day volume confirmation
# Long when price breaks above 20-period Donchian high + weekly pivot direction bullish + volume > 1.5x 20-day average
# Short when price breaks below 20-period Donchian low + weekly pivot direction bearish + volume > 1.5x 20-day average
# Exit when price crosses 5-period EMA in opposite direction
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-week pivot for trend direction (resists whipsaw) and 1-day volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_donchian20_1w_pivot_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for pivot trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using weekly OHLC)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Pivot trend: bullish if close > R1, bearish if close < S1
    pivot_bullish = close_1w > r1_1w
    pivot_bearish = close_1w < s1_1w
    
    # Align pivot signals to 6h timeframe
    pivot_bullish_aligned = align_htf_to_ltf(prices, df_1w, pivot_bullish.astype(float))
    pivot_bearish_aligned = align_htf_to_ltf(prices, df_1w, pivot_bearish.astype(float))
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 5-period EMA for exit
    ema_5 = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(pivot_bullish_aligned[i]) or 
            np.isnan(pivot_bearish_aligned[i]) or np.isnan(ema_5[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 5-period EMA
            elif close[i] < ema_5[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 5-period EMA
            elif close[i] > ema_5[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with pivot direction and volume confirmation
            # Volume filter: volume > 1.5x 20-day average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            
            # Long: price breaks above Donchian high + weekly pivot bullish + volume filter
            if close[i] > highest_high[i] and pivot_bullish_aligned[i] > 0.5 and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + weekly pivot bearish + volume filter
            elif close[i] < lowest_low[i] and pivot_bearish_aligned[i] > 0.5 and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals