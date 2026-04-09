#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels + 1d volume spike + 1w chop regime filter
# - Uses 1d Camarilla pivot levels (H3, L3) for mean reversion entries
# - Uses 1d volume spike (volume > 1.5x 20-period average) for confirmation
# - Uses 1w chop regime (choppiness index > 61.8) to filter ranging markets
# - Long when price touches L3 level from above in choppy market with volume spike
# - Short when price touches H3 level from below in choppy market with volume spike
# - Fixed position size 0.25 to control drawdown
# - ATR-based stoploss: exit when price moves 2.0x ATR against position
# - Works in both bull and bear markets by using mean reversion in ranging regimes
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_1w_camarilla_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla levels: H3, H4, L3, L4
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H3 = Pivot + Range * 1.1/2
    # L3 = Pivot - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3_1d = pivot_1d + range_1d * 1.1 / 2.0
    l3_1d = pivot_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ma_1d * 1.5)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Load 1w data ONCE before loop for chop regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Choppiness Index (14-period)
    # Chop = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_1w[0] = tr1_w[0]  # First bar
    
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero and log of zero
    denominator = atr_1w * 14
    chop_1w = np.where(
        (denominator > 0) & (sum_tr_14 > 0),
        100 * np.log10(sum_tr_14 / denominator) / np.log10(14),
        50.0  # neutral value when undefined
    )
    
    # Chop > 61.8 indicates ranging market (good for mean reversion)
    chop_regime_1w = chop_1w > 61.8
    chop_regime_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_regime_1w.astype(float))
    
    # Pre-compute ATR (14-period) for 12h timeframe stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_regime_1w_aligned[i]) or
            np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Regime filters: choppy market AND volume spike
        ranging_market = chop_regime_1w_aligned[i] > 0.5
        vol_spike = volume_spike_1d_aligned[i] > 0.5
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.0x ATR from highest high
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.0x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if ranging_market and vol_spike:
                # Long entry: price touches L3 level from above
                if low[i] <= l3_1d_aligned[i] and close[i] > l3_1d_aligned[i]:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price touches H3 level from below
                elif high[i] >= h3_1d_aligned[i] and close[i] < h3_1d_aligned[i]:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals