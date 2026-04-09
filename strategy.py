#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 12h volume spike + chop regime filter
# Camarilla levels provide precise intraday support/resistance from prior 12h range
# Volume spike confirms institutional participation in breakouts
# Choppiness regime filter adapts to market conditions: trend follow in trending, mean revert in ranging
# Works in bull/bear: regime filter prevents false signals, breakouts capture strong moves
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25-0.30

name = "4h_12h_camarilla_volume_chop_v1"
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
    
    # Load 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (based on prior 12h bar's high/low/close)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: based on previous period's range
    # H1 = close + 1.1*(high-low)/12, H2 = close + 1.1*(high-low)/6, H3 = close + 1.1*(high-low)/4
    # L1 = close - 1.1*(high-low)/12, L2 = close - 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/4
    range_12h = high_12h - low_12h
    H3 = close_12h + 1.1 * range_12h / 4
    H2 = close_12h + 1.1 * range_12h / 6
    H1 = close_12h + 1.1 * range_12h / 12
    L1 = close_12h - 1.1 * range_12h / 12
    L2 = close_12h - 1.1 * range_12h / 6
    L3 = close_12h - 1.1 * range_12h / 4
    
    # Calculate 12h average volume (20-period)
    volume_12h = df_12h['volume'].values
    volume_s_12h = pd.Series(volume_12h)
    avg_volume_12h = volume_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Choppiness Index (CHOP)
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
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
    
    atr_12h = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_12h - ll_12h
    chop_12h = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align 12h indicators to 4h timeframe (wait for 12h bar close)
    H3_aligned = align_htf_to_ltf(prices, df_12h, H3)
    H2_aligned = align_htf_to_ltf(prices, df_12h, H2)
    H1_aligned = align_htf_to_ltf(prices, df_12h, H1)
    L1_aligned = align_htf_to_ltf(prices, df_12h, L1)
    L2_aligned = align_htf_to_ltf(prices, df_12h, L2)
    L3_aligned = align_htf_to_ltf(prices, df_12h, L3)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(avg_volume_12h_aligned[i]) or np.isnan(chop_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 12h average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_12h_aligned[i]
        
        # Regime filter: CHOP < 38.2 = trending (follow breakout), CHOP > 61.8 = range (mean revert)
        trending_regime = chop_12h_aligned[i] < 38.2
        ranging_regime = chop_12h_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below H3 (Camarilla resistance) OR regime shifts to ranging
            if close[i] < H3_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above L3 (Camarilla support) OR regime shifts to ranging
            if close[i] > L3_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow breakout in trending regime
                if close[i] > H3_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < L3_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at Camarilla H3/L3 levels in ranging regime
                if close[i] < L3_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > H3_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals