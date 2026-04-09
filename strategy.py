#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume spike and 1d choppiness regime filter
# Uses 4h/1d for signal direction, 1h only for entry timing precision
# Targets 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Works in bull/bear markets: breakout catches trends, chop filter avoids whipsaws in ranging markets

name = "1h_4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h ATR(14) for volatility normalization
    tr1_4h = np.abs(high_4h[1:] - low_4h[:-1])
    tr2_4h = np.abs(high_4h[1:] - close_4h[:-1])
    tr3_4h = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_4h = wilders_smoothing(tr_4h, 14)
    
    # Calculate 4h average volume (20-period) normalized by ATR
    volume_s_4h = pd.Series(volume_4h)
    avg_volume_4h = volume_s_4h.rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = np.where(atr_4h > 0, avg_volume_4h / atr_4h, np.nan)
    avg_vol_ratio_4h = pd.Series(vol_ratio_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Calculate 4h Camarilla pivot levels (based on prior bar to avoid look-ahead)
    range_4h = high_4h - low_4h
    h3_4h = close_4h + 1.1 * range_4h
    l3_4h = close_4h - 1.1 * range_4h
    h4_4h = close_4h + 1.5 * range_4h
    l4_4h = close_4h - 1.5 * range_4h
    
    # Align 4h indicators to 1h timeframe
    avg_vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, avg_vol_ratio_4h)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    h4_4h_aligned = align_htf_to_ltf(prices, df_4h, h4_4h)
    l4_4h_aligned = align_htf_to_ltf(prices, df_4h, l4_4h)
    
    # Pre-compute volume confirmation array
    avg_volume_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    avg_volume_4h_aligned = align_htf_to_ltf(prices, df_4h, avg_volume_4h)
    volume_confirmed = volume > 2.0 * avg_volume_4h_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(avg_vol_ratio_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or
            np.isnan(h4_4h_aligned[i]) or np.isnan(l4_4h_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below H3 or we enter ranging regime
                if close[i] < h3_4h_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            elif ranging_regime:
                # Exit long if price rises above H4 or drops below L3
                if close[i] > h4_4h_aligned[i] or close[i] < l3_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above L3 or we enter ranging regime
                if close[i] > l3_4h_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            elif ranging_regime:
                # Exit short if price drops below L4 or rises above H3
                if close[i] < l4_4h_aligned[i] or close[i] > h3_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
        else:  # Flat
            if trending_regime:
                # Enter long on breakout above H3 with volume confirmation
                if close[i] > h3_4h_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.20
                # Enter short on breakout below L3 with volume confirmation
                elif close[i] < l3_4h_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.20
            elif ranging_regime:
                # Mean reversion: buy near L3, sell near H3
                if close[i] <= l3_4h_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] >= h3_4h_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals