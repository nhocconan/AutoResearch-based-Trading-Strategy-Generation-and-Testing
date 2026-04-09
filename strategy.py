#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR volume confirmation + 1d chop regime filter
# Donchian breakout captures strong momentum moves in both bull and bear markets
# 1d ATR-normalized volume confirms breakout authenticity (adjusts for volatility)
# Choppiness index regime filter: CHOP < 38.2 = trending (follow breakout), CHOP > 61.8 = range (mean revert)
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years.

name = "4h_1d_donchian_atr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volume normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's ATR smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # 1d ATR-normalized volume (volume / ATR) - 20 period average
    volume_1d = df_1d['volume'].values
    vol_norm_1d = volume_1d / (atr_1d + 1e-10)  # avoid division by zero
    avg_vol_norm_1d = pd.Series(vol_norm_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d Choppiness Index (CHOP)
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Align 1d indicators to 4h timeframe
    avg_vol_norm_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_norm_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(avg_vol_norm_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h ATR-normalized volume > 1.3x 1d average
        vol_norm_current = volume[i] / (atr_1d[-1] if len(atr_1d) > 0 else 1e-10)  # approximate current ATR
        # Better: use rolling ATR on 4h for current vol norm
        if i >= 14:
            # Calculate 4h ATR for current bar
            tr_4h = []
            for j in range(max(0, i-13), i+1):
                if j == 0:
                    tr_4h.append(high[j] - low[j])
                else:
                    tr1 = abs(high[j] - low[j])
                    tr2 = abs(high[j] - close[j-1])
                    tr3 = abs(low[j] - close[j-1])
                    tr_4h.append(max(tr1, tr2, tr3))
            atr_4h_current = np.mean(tr_4h[-14:]) if len(tr_4h) >= 14 else np.mean(tr_4h)
            vol_norm_current = volume[i] / (atr_4h_current + 1e-10)
        else:
            vol_norm_current = volume[i] / (np.mean(high-low) + 1e-10)
            
        volume_confirmed = vol_norm_current > 1.3 * avg_vol_norm_1d_aligned[i]
        
        # Regime filter
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if close[i] < lowest_low[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if close[i] > highest_high[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Follow breakout in trending regime
                if close[i] > highest_high[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lowest_low[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at Donchian bands in ranging regime
                if close[i] < lowest_low[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > highest_high[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals