#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels with 1d trend filter and volume confirmation.
# Fade at R3/S3 (mean reversion) during choppy markets, breakout continuation at R4/S4 during trending markets.
# Uses 1d ADX > 25 to identify trending regimes and 1d close > open for trend direction.
# Volume > 1.5x 20-period average confirms institutional participation.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "6h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    # Use previous day's OHLC for today's pivot levels (avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    # Avoid division by zero
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    # Pivot point
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Camarilla levels
    r3 = pp + (range_val * 1.1 / 2.0)
    s3 = pp - (range_val * 1.1 / 2.0)
    r4 = pp + (range_val * 1.1)
    s4 = pp - (range_val * 1.1)
    
    # Align levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d trend filter: ADX > 25 indicates trending market
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (14-period)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d trend direction: bullish if close > open
    daily_bullish = df_1d['close'] > df_1d['open']
    daily_bearish = df_1d['close'] < df_1d['open']
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.values)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.values)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below S3 (mean reversion) or strong adverse move below S4
            if (low[i] <= s3_aligned[i] and adx_aligned[i] < 25) or low[i] <= s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above R3 (mean reversion) or strong adverse move above R4
            if (high[i] >= r3_aligned[i] and adx_aligned[i] < 25) or high[i] >= r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter:
                # Mean reversion fade at R3/S3 in choppy markets (ADX < 25)
                if adx_aligned[i] < 25:
                    # Long at S3 support
                    if low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    # Short at R3 resistance
                    elif high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                # Breakout continuation at R4/S4 in trending markets (ADX >= 25)
                else:
                    # Long breakout above R4 with bullish bias
                    if high[i] > r4_aligned[i] and daily_bullish_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    # Short breakdown below S4 with bearish bias
                    elif low[i] < s4_aligned[i] and daily_bearish_aligned[i]:
                        signals[i] = -0.25
                        position = -1
    
    return signals