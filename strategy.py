#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and choppiness regime filter
# In trending regimes (CHOP < 38.2): breakout above/below Donchian(20) with volume confirmation
# In ranging regimes (CHOP > 61.8): mean reversion at Donchian(20) upper/lower bands with volume confirmation
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: breakout catches trends, chop filter avoids whipsaws in ranging markets

name = "4h_12h_donchian_breakout_volume_chop_v2"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ATR(14) for volatility normalization
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
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
    
    atr_12h = wilders_smoothing(tr, 14)
    
    # Calculate 12h average volume (20-period) normalized by ATR
    volume_s_12h = pd.Series(volume_12h)
    avg_volume_12h = volume_s_12h.rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = np.where(atr_12h > 0, avg_volume_12h / atr_12h, np.nan)
    avg_vol_ratio_12h = pd.Series(vol_ratio_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Choppiness Index (CHOP)
    hh_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_12h - ll_12h
    chop_12h = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Calculate 4h Donchian(20) channels (based on prior period to avoid look-ahead)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 12h indicators to 4h timeframe
    avg_vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_vol_ratio_12h)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Pre-compute volume confirmation array
    avg_volume_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    volume_confirmed = volume > 2.0 * avg_volume_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(avg_vol_ratio_12h_aligned[i]) or np.isnan(chop_12h_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = chop_12h_aligned[i] < 38.2
        ranging_regime = chop_12h_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below lower Donchian band or we enter ranging regime
                if close[i] < lowest_20[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price rises above upper Donchian band or drops below lower band
                if close[i] > highest_20[i] or close[i] < lowest_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above upper Donchian band or we enter ranging regime
                if close[i] > highest_20[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price drops below lower Donchian band or rises above upper band
                if close[i] < lowest_20[i] or close[i] > highest_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above upper Donchian band with volume confirmation
                if close[i] > highest_20[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on breakout below lower Donchian band with volume confirmation
                elif close[i] < lowest_20[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion: buy near lower band, sell near upper band
                if close[i] <= lowest_20[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= highest_20[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals