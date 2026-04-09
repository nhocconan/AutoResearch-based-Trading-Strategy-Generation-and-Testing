#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout + 1d Regime Filter (ADX) + Volume Spike
# Uses Bollinger Bands (20,2) on 6h to detect low volatility squeezes (bandwidth < 20th percentile)
# Breakout triggers when price closes outside bands with volume > 2x 20-period average
# 1d ADX > 25 confirms trending regime to avoid false breakouts in ranging markets
# Works in bull/bear: ADX filter ensures we only trade breakouts in trending conditions
# Target: 80-160 total trades over 4 years (20-40/year) with discrete sizing 0.25

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
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX (14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # Rest: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * std_20)
    lower_band = sma_20 - (bb_std * std_20)
    
    # Bollinger Band Width (for squeeze detection)
    bb_width = (upper_band - lower_band) / sma_20
    # 20th percentile of BB width as squeeze threshold (lookback 50 periods)
    bb_width_percentile = np.full(n, np.nan)
    for i in range(50, n):
        if i >= 50:
            bb_width_percentile[i] = np.nanpercentile(bb_width[i-50:i], 20)
    
    # Volume confirmation: 20-period average
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d ADX > 25 (trending market)
        trending_regime = adx_1d_aligned[i] > 25
        
        # Squeeze condition: BB width < 20th percentile (low volatility)
        squeeze_condition = bb_width[i] < bb_width_percentile[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price re-enters Bollinger Bands (mean reversion of squeeze)
            if lower_band[i] <= close[i] <= upper_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Bollinger Bands
            if lower_band[i] <= close[i] <= upper_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Bollinger Band breakout during squeeze in trending regime
            if trending_regime and squeeze_condition and volume_confirmed:
                # Long breakout: price closes above upper band
                if close[i] > upper_band[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below lower band
                elif close[i] < lower_band[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals