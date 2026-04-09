#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels + volume confirmation + choppiness regime filter
# Long when price breaks above H3 level with volume confirmation in low-chop regime (trending)
# Short when price breaks below L3 level with volume confirmation in low-chop regime
# Uses discrete position sizing 0.25 to target ~25-40 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows trends, chop filter avoids ranging markets

name = "4h_1d_camarilla_breakout_chop_v1"
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
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    camarilla_h3 = close_1d + 1.25 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.25 * (high_1d - low_1d)
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute volume confirmation: current 4h volume > 2.0x average 4h volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 2.0 * vol_ma_20
    
    # Pre-compute choppiness regime filter: CHOP(14) < 38.2 = trending regime
    # Chop = 100 * log10(sum(ATR(1)/n) / (log10(n) * (highest_high - lowest_low)))
    atr_1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_1[0] = high[0] - low[0]  # first bar
    tr_sum = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr_sum / (np.log10(14) * (highest_high - lowest_low)))
    chop = np.where((highest_high - lowest_low) == 0, 100, chop)  # avoid div by zero
    chop = np.where(np.isnan(chop), 100, chop)  # fill NaN with high chop (ranging)
    low_chop_regime = chop < 38.2  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price falls below L3 level
            if close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above H3 level
            if close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on Camarilla breakout with volume confirmation in trending regime
            if (close[i] > camarilla_h3_aligned[i] and volume_confirmed[i] and low_chop_regime[i]):
                position = 1
                signals[i] = 0.25
            elif (close[i] < camarilla_l3_aligned[i] and volume_confirmed[i] and low_chop_regime[i]):
                position = -1
                signals[i] = -0.25
    
    return signals