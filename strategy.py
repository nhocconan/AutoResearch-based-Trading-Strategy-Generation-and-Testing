#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume confirmation + chop regime filter
# Camarilla pivots from 1d provide strong intraday support/resistance levels
# Breakout above H3 or below L3 with volume confirmation captures strong moves
# Choppiness index regime filter: CHOP > 61.8 = range (mean revert at bands), CHOP < 38.2 = trending (follow breakout)
# Works in bull/bear: regime filter adapts, breakout captures strong moves
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_camarilla_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla, volume and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d typical price for Camarilla
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    tp_1d = typical_price_1d.values
    
    # Calculate 1d Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # H2 = close + 1.166 * (high - low)
    # H1 = close + 1.0833 * (high - low)
    # L1 = close - 1.0833 * (high - low)
    # L2 = close - 1.166 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # L4 = close - 1.5 * (high - low)
    
    range_1d = high_1d - low_1d
    h4_1d = close_1d + 1.5 * range_1d
    h3_1d = close_1d + 1.25 * range_1d
    h2_1d = close_1d + 1.166 * range_1d
    h1_1d = close_1d + 1.0833 * range_1d
    l1_1d = close_1d - 1.0833 * range_1d
    l2_1d = close_1d - 1.166 * range_1d
    l3_1d = close_1d - 1.25 * range_1d
    l4_1d = close_1d - 1.5 * range_1d
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
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
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar close)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Regime filter: CHOP < 38.2 = trending (follow breakout), CHOP > 61.8 = range (mean revert)
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR regime shifts to ranging
            if close[i] < l3_1d_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR regime shifts to ranging
            if close[i] > h3_1d_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow breakout in trending regime
                if close[i] > h3_1d_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < l3_1d_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at H3/L3 in ranging regime
                if close[i] < l3_1d_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > h3_1d_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals