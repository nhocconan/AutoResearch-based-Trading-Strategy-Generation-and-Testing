#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout + 1d ADX Trend Filter + Volume Spike
# Uses Bollinger Bands width percentile to detect low volatility squeezes (BBW < 20th percentile)
# Breakout triggered when price closes outside BB(20,2) AND volume > 2x 20-period average
# 1d ADX > 25 confirms trending regime to avoid false breakouts in ranging markets
# Works in bull/bear: ADX filter ensures we only trade breakouts in trending conditions
# Target: 80-120 total trades over 4 years (20-30/year) with discrete sizing 0.25

name = "6h_1d_bb_squeeze_breakout_adx_volume_v1"
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
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
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
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]),
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_smoothed = np.full_like(tr, np.nan)
    dm_plus_smoothed = np.full_like(dm_plus, np.nan)
    dm_minus_smoothed = np.full_like(dm_minus, np.nan)
    
    # Initial smoothed values (simple average)
    if len(tr) >= tr_period:
        tr_smoothed[tr_period-1] = np.nanmean(tr[1:tr_period])
        dm_plus_smoothed[tr_period-1] = np.nanmean(dm_plus[1:tr_period])
        dm_minus_smoothed[tr_period-1] = np.nanmean(dm_minus[1:tr_period])
    
    # Wilder smoothing
    for i in range(tr_period, len(tr)):
        tr_smoothed[i] = tr_smoothed[i-1] - (tr_smoothed[i-1] / tr_period) + tr[i]
        dm_plus_smoothed[i] = dm_plus_smoothed[i-1] - (dm_plus_smoothed[i-1] / tr_period) + dm_plus[i]
        dm_minus_smoothed[i] = dm_minus_smoothed[i-1] - (dm_minus_smoothed[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full_like(tr, np.nan)
    di_minus = np.full_like(tr, np.nan)
    for i in range(tr_period, len(tr)):
        if tr_smoothed[i] != 0:
            di_plus[i] = 100 * dm_plus_smoothed[i] / tr_smoothed[i]
            di_minus[i] = 100 * dm_minus_smoothed[i] / tr_smoothed[i]
    
    # DX and ADX
    dx = np.full_like(tr, np.nan)
    for i in range(tr_period, len(tr)):
        if di_plus[i] + di_minus[i] != 0:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    adx = np.full_like(tr, np.nan)
    for i in range(2*tr_period-1, len(tr)):
        if i == 2*tr_period-1:
            adx[i] = np.nanmean(dx[tr_period:i+1])
        else:
            adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    adx_1d = adx
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std_dev)
    lower_band = sma - (bb_std * std_dev)
    bb_width = (upper_band - lower_band) / sma * 100  # Percentage width
    
    # Calculate BB width percentile rank (lookback 50 periods)
    bb_width_percentile = np.full(n, np.nan)
    lookback = 50
    for i in range(lookback, n):
        if not np.isnan(bb_width[i-lookback:i+1]).any():
            bb_width_percentile[i] = (np.sum(bb_width[i-lookback:i] <= bb_width[i]) / lookback) * 100
    
    # Calculate 20-period average volume for volume spike
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bb_width_percentile[i]) or
            np.isnan(avg_volume[i]) or np.isnan(sma[i]) or np.isnan(std_dev[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        bb_squeeze = bb_width_percentile[i] < 20  # Low volatility squeeze
        volume_spike = volume[i] > 2.0 * avg_volume[i]  # Volume confirmation
        adx_trending = adx_1d_aligned[i] > 25  # Trending regime on 1d
        
        if position == 1:  # Long position
            # Exit: price closes back inside Bollinger Bands OR ADX drops below 20
            if close[i] < upper_band[i] and close[i] > lower_band[i] or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside Bollinger Bands OR ADX drops below 20
            if close[i] < upper_band[i] and close[i] > lower_band[i] or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Bollinger Band breakout with volume spike in trending regime
            if bb_squeeze and volume_spike and adx_trending:
                # Long breakout: price closes above upper Bollinger Band
                if close[i] > upper_band[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below lower Bollinger Band
                elif close[i] < lower_band[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals