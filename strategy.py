#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d data + volume spike + choppiness regime filter
# Uses 1d Camarilla levels (H3, L3) as breakout triggers, volume > 2x average for confirmation,
# and choppiness index (1d) > 61.8 to ensure ranging market for mean-reversion at pivot touches.
# Works in bull/bear: choppiness filter avoids trending markets where pivots fail,
# volume confirmation reduces false breakouts, Camarilla provides mathematically derived support/resistance.
# Target: 20-40 trades/year (75-150 over 4 years) to avoid fee drag.

name = "4h_1d_camarilla_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla levels and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # H3 = close + 1.1*(high - low)/2, L3 = close - 1.1*(high - low)/2
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First value will be NaN due to roll, handled by min_periods in rolling if needed
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align 1d Camarilla levels to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR1) / (n * log(n+1))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (14 * log10(15))) / log10(14)
    # We'll use common approximation: CHOP = 100 * log10(ATR(14) / (TrueRange_sum)) / log10(14)
    # Actually standard formula: CHOP = 100 * LOG10(ATR(14) / (SUM(TrueRange,14))) / LOG10(14)
    # But we'll use Wilder's ATR and True Range sum over 14 periods
    tr1 = np.maximum(high_1d, np.roll(close_1d, 1)) - np.minimum(low_1d, np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean()
    tr_sum14 = pd.Series(tr1).rolling(window=14, min_periods=14).sum()
    # Avoid division by zero
    chop = 100 * np.log10(tr_sum14 / (atr1 * 14 + 1e-10)) / np.log10(14)
    chop_values = chop.values
    
    # Align 1d Choppiness to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Calculate 20-period average volume for volume spike confirmation (4h volume)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or
            np.isnan(chop_4h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * avg_volume[i]
        # Choppiness regime: CHOP > 61.8 indicates ranging market (good for mean reversion at pivots)
        ranging_market = chop_4h[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price reaches Camarilla H3 (profit target) OR breaks below L3 (stop loss)
            if close[i] >= h3_4h[i] or close[i] <= l3_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Camarilla L3 (profit target) OR breaks above H3 (stop loss)
            if close[i] <= l3_4h[i] or close[i] >= h3_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and ranging market filter
            if volume_confirm and ranging_market:
                # Long entry: price touches Camarilla L3 from above (support bounce)
                if low[i] <= l3_4h[i] and close[i] > l3_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches Camarilla H3 from below (resistance rejection)
                elif high[i] >= h3_4h[i] and close[i] < h3_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals