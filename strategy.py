#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation
# Enter long when BB width < 20th percentile (squeeze) AND price breaks above upper band AND 1d ADX > 25 AND volume > 1.5x average
# Enter short when BB width < 20th percentile (squeeze) AND price breaks below lower band AND 1d ADX > 25 AND volume > 1.5x average
# Exit when price returns to middle band (mean reversion) or BB width expands above 50th percentile (squeeze end)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 6h.
# Bollinger squeeze captures low volatility breakouts that often trend well. 1d ADX filter ensures we only trade in trending regimes on higher timeframe, reducing false breakouts in ranging markets.
# Volume confirmation adds conviction to breakouts.

name = "6h_BBSqueeze_1dADX_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smooth = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 27:  # Need 14 for DX + 14 for ADX smoothing
        adx[26] = np.nanmean(dx[14:27])  # First ADX is average of first 14 DX
        for i in range(27, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Bollinger Bands (20, 2) on 6h data
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Percentile of BB width (lookback 50 periods for regime)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(50, len(bb_width)):
        bb_width_percentile[i] = (np.sum(bb_width[i-50:i] <= bb_width[i]) / 50) * 100
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for BB percentile and bands
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(bb_width_percentile[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        adx_val = adx_1d_aligned[i]
        width_percentile = bb_width_percentile[i]
        curr_close = close[i]
        bb_mid = bb_middle[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when BB squeeze (width < 20th percentile) AND price breaks above upper band AND 1d ADX > 25 AND volume confirmation
            if (width_percentile < 20 and curr_close > bb_up and adx_val > 25 and vol_conf):
                signals[i] = 0.25
                position = 1
            # Short when BB squeeze (width < 20th percentile) AND price breaks below lower band AND 1d ADX > 25 AND volume confirmation
            elif (width_percentile < 20 and curr_close < bb_low and adx_val > 25 and vol_conf):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to middle band OR squeeze ends (width > 50th percentile)
            if curr_close <= bb_mid or width_percentile > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to middle band OR squeeze ends (width > 50th percentile)
            if curr_close >= bb_mid or width_percentile > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals