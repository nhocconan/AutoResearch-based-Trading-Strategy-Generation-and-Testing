#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R for momentum and 1w ATR percentile for volatility regime
# - Uses 1w ATR percentile to identify low volatility regimes (squeeze) - contraction phase
# - Uses 1d Williams %R < -80 for oversold and > -20 for overbought conditions
# - Enters long when price closes above 1d high with Williams %R crossing above -80 in low vol
# - Enters short when price closes below 1d low with Williams %R crossing below -20 in low vol
# - Exits when Williams %R returns to neutral range (-80 to -20) or volatility expands
# - Designed to capture mean reversion bursts after weekly consolidation with momentum confirmation
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_1wATRPercentile_1dWilliamsR_MeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and high/low levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Get 1w data for ATR calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1w ATR (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    
    # Calculate 1w ATR percentile rank (lookback 50 periods)
    atr_series = pd.Series(atr_14)
    atr_percentile = atr_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Get 1d high and low for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align 1d indicators to 6h timeframe
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    high_1d_6h = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_6h = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Align 1w ATR percentile to 6h timeframe
    atr_percentile_6h = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Volume filter (6h timeframe) - moderate threshold to avoid overtrading
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_avg = volume > vol_ma_20  # Above average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(high_1d_6h[i]) or np.isnan(low_1d_6h[i]) or
            np.isnan(atr_percentile_6h[i]) or np.isnan(volume_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for low volatility regime (ATR < 30th percentile) 
            low_vol_regime = atr_percentile_6h[i] < 30
            
            if low_vol_regime and volume_avg[i]:
                # Long: Williams %R crosses above -80 (oversold recovery) and price > 1d high
                if williams_r_6h[i] > -80 and williams_r_6h[i-1] <= -80 and close[i] > high_1d_6h[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 (overbought decline) and price < 1d low
                elif williams_r_6h[i] < -20 and williams_r_6h[i-1] >= -20 and close[i] < low_1d_6h[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or volatility expands (ATR > 70th percentile)
            if williams_r_6h[i] > -50 or atr_percentile_6h[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or volatility expands (ATR > 70th percentile)
            if williams_r_6h[i] < -50 or atr_percentile_6h[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals