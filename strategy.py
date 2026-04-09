#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels + volume confirmation + choppiness regime filter
# Camarilla pivots provide key support/resistance levels based on prior 1d range
# Long when price breaks above H3 level with volume confirmation in trending regime (CHOP < 38.2)
# Short when price breaks below L3 level with volume confirmation in trending regime (CHOP < 38.2)
# Uses discrete position sizing 0.25 to target ~20-40 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows trends, chop filter avoids range-bound whipsaws

name = "12h_1d_camarilla_breakout_chop_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    # H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low)
    # L3 = close - 1.25*(high-low), L4 = close - 1.5*(high-low)
    camarilla_h4_1d = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_h3_1d = close_1d + 1.25 * (high_1d - low_1d)
    camarilla_l3_1d = close_1d - 1.25 * (high_1d - low_1d)
    camarilla_l4_1d = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align 1d indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Pre-compute 12h ATR(14) for stoploss and choppiness
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 12h Choppiness Index (CHOP) - higher = more choppy/ranging
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    chop = np.where((max_high - min_low) > 0, chop, 50.0)  # neutral when no range
    
    # Pre-compute volume confirmation: current 12h volume > 1.5x average 12h volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(chop[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        if chop[i] >= 38.2:
            # In choppy/ranging market, flatten position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price falls below L3 level or ATR-based stoploss
            if close[i] < camarilla_l3_aligned[i] or close[i] < (prices['high'].rolling(3).max().iloc[i] - 2.0 * atr_14[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above H3 level or ATR-based stoploss
            if close[i] > camarilla_h3_aligned[i] or close[i] > (prices['low'].rolling(3).min().iloc[i] + 2.0 * atr_14[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on Camarilla breakout with volume confirmation in trending regime
            if close[i] > camarilla_h3_aligned[i] and volume_confirmed[i]:
                position = 1
                signals[i] = 0.25
            elif close[i] < camarilla_l3_aligned[i] and volume_confirmed[i]:
                position = -1
                signals[i] = -0.25
    
    return signals