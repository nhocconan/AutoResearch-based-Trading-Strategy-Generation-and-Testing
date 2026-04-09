#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + 1w volume confirmation + chop regime filter
# Camarilla pivots provide intraday support/resistance levels that work in ranging markets
# 1w volume spike confirms institutional participation (avoids false breakouts)
# Choppiness index regime filter: CHOP > 61.8 = range (mean revert at Camarilla H3/L3), CHOP < 38.2 = trending (follow breakout H4/L4)
# Works in bull/bear: regime filter adapts, Camarilla levels provide structure in both markets
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1w_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # H4 = PP + range * 1.1/2
    # L4 = PP - range * 1.1/2
    # H3 = PP + range * 1.1/4
    # L3 = PP - range * 1.1/4
    camarilla_h4 = pp_1d + range_1d * 1.1 / 2.0
    camarilla_l4 = pp_1d - range_1d * 1.1 / 2.0
    camarilla_h3 = pp_1d + range_1d * 1.1 / 4.0
    camarilla_l3 = pp_1d - range_1d * 1.1 / 4.0
    
    # Calculate 1w volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    volume_s_1w = pd.Series(volume_1w)
    avg_volume_1w = volume_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align HTF indicators to 12h timeframe (wait for bar close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    avg_volume_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_1w)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_volume_1w_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 1w average volume
        volume_confirmed = volume[i] > 1.3 * avg_volume_1w_aligned[i]
        
        # Regime filter
        trending_regime = chop_1d_aligned[i] < 40   # Strong trend
        ranging_regime = chop_1d_aligned[i] > 60    # Strong range
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR regime shifts to ranging
            if close[i] < camarilla_l3_aligned[i] or (ranging_regime and not trending_regime):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR regime shifts to ranging
            if close[i] > camarilla_h3_aligned[i] or (ranging_regime and not trending_regime):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime and volume_confirmed:
                # Follow breakout in trending regime
                if close[i] > camarilla_h4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < camarilla_l4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime and volume_confirmed:
                # Mean revert at Camarilla H3/L3 in ranging regime
                if close[i] < camarilla_l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > camarilla_h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals