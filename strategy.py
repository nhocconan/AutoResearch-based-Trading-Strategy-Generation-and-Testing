#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ADX for trend strength and 1w Bollinger Band width for volatility regime
# - Uses 1w Bollinger Band width percentile to identify low volatility regimes (squeeze) - contrarian indicator
# - Uses 1d ADX > 25 to confirm trend strength for breakout direction
# - Enters long when price breaks above 1d high with volume spike in low vol + strong trend
# - Enters short when price breaks below 1d low with volume spike in low vol + strong trend
# - Exits when price crosses back below/above 1d close or volatility expands (BB width > 80th percentile)
# - Designed to capture volatility breakouts after weekly consolidation with daily trend confirmation
# - Target: 80-160 total trades over 4 years (20-40/year) with 0.25 position sizing

name = "4h_1wBBWidth_1dADX_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for 1d high/low and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get 1w data for Bollinger Band width calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d high and low for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate 1w Bollinger Bands (20, 2)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Middle band (SMA20)
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    std_dev = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    upper_bb = sma_20 + (2 * std_dev)
    lower_bb = sma_20 - (2 * std_dev)
    # Bollinger Band width
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Calculate 1w BB width percentile rank (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align 1d indicators to 4h timeframe
    high_1d_4h = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_4h = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_4h = align_htf_to_ltf(prices, df_1d, close_1d)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Align 1w BB width percentile to 4h timeframe
    bb_width_percentile_4h = align_htf_to_ltf(prices, df_1w, bb_width_percentile)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(high_1d_4h[i]) or np.isnan(low_1d_4h[i]) or np.isnan(close_1d_4h[i]) or
            np.isnan(adx_4h[i]) or np.isnan(bb_width_percentile_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for low volatility regime (BB width < 20th percentile) and strong trend (ADX > 25)
            low_vol_regime = bb_width_percentile_4h[i] < 20
            strong_trend = adx_4h[i] > 25
            
            if low_vol_regime and strong_trend:
                # Long: price breaks above 1d high with volume spike
                if close[i] > high_1d_4h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 1d low with volume spike
                elif close[i] < low_1d_4h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below 1d close OR volatility expands (BB width > 80th percentile)
            if close[i] < close_1d_4h[i] or bb_width_percentile_4h[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d close OR volatility expands (BB width > 80th percentile)
            if close[i] > close_1d_4h[i] or bb_width_percentile_4h[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals