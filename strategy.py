#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with weekly Bollinger Band squeeze breakout + volume confirmation + ADX trend filter.
# Weekly Bollinger Band squeeze indicates low volatility and potential for explosive moves.
# Breakout from squeeze with volume surge captures institutional participation.
# ADX filter ensures trades occur in trending markets, avoiding chop.
# Works in bull/bear by following strong trends from volatility contractions.
# Target: 30-100 total trades over 4 years (7-25/year). Size: 0.25.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Weekly Bollinger Bands (20, 2) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Bollinger Bands
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = (bb_upper - bb_lower) / sma_20  # Normalized width
    
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    bb_width_aligned = align_htf_to_ltf(prices, df_1w, bb_width)
    
    # === Weekly ADX (14) for trend strength ===
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = dm_minus[0] = 0
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    di_plus = 100 * dm_plus_smooth / atr_14
    di_minus = 100 * dm_minus_smooth / atr_14
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx_14 = wilders_smoothing(dx, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14)
    
    # === Daily volume for surge confirmation ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for weekly indicators
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # Bollinger Band squeeze: width below 20th percentile of last 50 weeks
        if i >= 50:
            width_history = bb_width_aligned[max(0, i-50):i]
            width_percentile = np.percentile(width_history, 20)
            squeeze = bb_width_aligned[i] <= width_percentile
        else:
            squeeze = False
        
        # Volume surge: current 1d volume > 1.5x 20-period average
        df_1d_current = get_htf_data(prices, '1d')
        vol_1d_current = df_1d_current['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d_current, vol_1d_current)
        vol_surge = vol_1d_aligned[i] > vol_ma_20_aligned[i] * 1.5
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1w_aligned[i] > 25.0
        
        # Entry logic: only enter when flat and squeeze condition met
        if position == 0 and squeeze:
            # Long: Price breaks above weekly Bollinger upper band + volume surge + trending
            if price > bb_upper_aligned[i] and vol_surge and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below weekly Bollinger lower band + volume surge + trending
            elif price < bb_lower_aligned[i] and vol_surge and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite breakout
        elif position == 1:
            # Exit long if price breaks below weekly Bollinger lower band
            if price < bb_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above weekly Bollinger upper band
            if price > bb_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBBSqueeze_Breakout_Volume1.5x_ADX25_TrendFilter"
timeframe = "1d"
leverage = 1.0