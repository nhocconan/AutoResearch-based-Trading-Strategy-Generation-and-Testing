#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter
# In trending regimes (CHOP < 38.2): breakout above/below Donchian(20) with volume confirmation
# In ranging regimes (CHOP > 61.8): fade Donchian(20) touches with volume confirmation
# Uses discrete position sizing 0.30 to limit trades to ~20-50/year and reduce fee drag
# Works in bull/bear markets: breakout catches trends, chop filter avoids whipsaws in ranging markets

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for volatility normalization
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
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
    
    # Calculate 1d average volume (20-period) normalized by ATR
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.where(atr_1d > 0, avg_volume_1d / atr_1d, np.nan)
    avg_vol_ratio_1d = pd.Series(vol_ratio_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Calculate 4h Donchian(20) channels (based on prior 20 bars to avoid look-ahead)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 1d indicators to 4h timeframe
    avg_vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_ratio_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute volume confirmation array
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    volume_confirmed = volume > 2.0 * avg_volume_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(avg_vol_ratio_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below lowest_20 or we enter ranging regime
                if close[i] < lowest_20[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            elif ranging_regime:
                # Exit long if price rises above highest_20 or drops below lowest_20
                if close[i] > highest_20[i] or close[i] < lowest_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above highest_20 or we enter ranging regime
                if close[i] > highest_20[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
            elif ranging_regime:
                # Exit short if price drops below lowest_20 or rises above highest_20
                if close[i] < lowest_20[i] or close[i] > highest_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above highest_20 with volume confirmation
                if close[i] > highest_20[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.30
                # Enter short on breakout below lowest_20 with volume confirmation
                elif close[i] < lowest_20[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.30
            elif ranging_regime:
                # Mean reversion: sell near highest_20, buy near lowest_20
                if close[i] >= highest_20[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.30
                elif close[i] <= lowest_20[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.30
    
    return signals