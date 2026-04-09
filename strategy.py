#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels + volume spike + choppiness regime filter
# Long when price touches Camarilla L3 support with volume spike in choppy market (CHOP > 61.8)
# Short when price touches Camarilla H3 resistance with volume spike in choppy market
# Uses discrete position sizing 0.25 to target ~20-40 trades/year
# Works in bull/bear markets: mean reversion in range, avoids trending markets via chop filter

name = "4h_1d_camarilla_pivot_volume_chop_v4"
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
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), etc.
    # We use H3 (resistance) and L3 (support) for entries
    hl_range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.0 * hl_range_1d  # H3 resistance
    camarilla_l3 = close_1d - 1.0 * hl_range_1d  # L3 support
    
    # Calculate 1d average volume (20-period)
    vol_s_1d = pd.Series(df_1d['volume'].values)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate 4h Choppiness Index (14-period) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom_safe = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10(np.sum(tr) / chop_denom_safe) / np.log10(14) if False else \
           100 * np.log10(pd.Series(tr).rolling(window=14, min_periods=14).sum().values / chop_denom_safe) / np.log10(14)
    chop = np.where(chop_denom == 0, 50, chop)  # set to neutral when range is zero
    
    # Pre-compute volume confirmation (4h)
    vol_s = pd.Series(volume)
    vol_ma_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20  # volume > 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(chop[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in choppy/range markets (CHOP > 61.8)
        if chop[i] <= 61.8:
            # In trending markets, stay flat to avoid whipsaws
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: volume spike
        if not volume_spike[i]:
            # No volume spike, stay flat or maintain position weakly
            if position != 0:
                # Hold position but reduce size slightly during low volume
                signals[i] = 0.15 if position == 1 else -0.15
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price moves above L3 (mean reversion complete) or below H3 (stop)
            if close[i] > camarilla_l3_aligned[i] or close[i] < camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price moves below H3 (mean reversion complete) or above L3 (stop)
            if close[i] < camarilla_h3_aligned[i] or close[i] > camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion: enter at Camarilla levels with volume spike in choppy market
            # Long at L3 support, Short at H3 resistance
            if abs(close[i] - camarilla_l3_aligned[i]) < 0.001 * camarilla_l3_aligned[i]:  # near L3
                position = 1
                signals[i] = 0.25
            elif abs(close[i] - camarilla_h3_aligned[i]) < 0.001 * camarilla_h3_aligned[i]:  # near H3
                position = -1
                signals[i] = -0.25
    
    return signals