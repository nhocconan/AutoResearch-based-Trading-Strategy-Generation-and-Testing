#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Primary: 4h price touching Camarilla H3/L3 levels from prior 1d range for mean reversion
# - HTF: 1d volume spike > 1.5x 20-period MA for confirmation
# - HTF: 1d choppiness index > 61.8 (range regime) to avoid trending markets
# - Long: price <= L3 (support) + volume spike + chop > 61.8
# - Short: price >= H3 (resistance) + volume spike + chop > 61.8
# - Exit: price crosses Camarilla H4/L4 (extreme levels) or opposite H3/L3 touch
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: chop filter ensures ranging markets, volume spike validates breakouts
# - Target: 75-200 trades over 4 years (19-50/year)

name = "4h_1d_camarilla_chop_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (based on prior day)
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    daily_range = high_1d - low_1d
    H3 = close_1d + 1.1 * daily_range
    L3 = close_1d - 1.1 * daily_range
    H4 = close_1d + 1.5 * daily_range
    L4 = close_1d - 1.5 * daily_range
    
    # Align Camarilla levels to 4h (prior day's levels available at 4h open)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Calculate 1d volume MA(20) for spike detection
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d choppiness index (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (max(high14)-min(low14))) / log10(14)
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.maximum(np.abs(low_1d - np.roll(close_1d, 1)), tr1)
    tr1[0] = high_1d[0] - low_1d[0]  # First TR
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop_numerator = np.sum(atr14[-14:]) if len(atr14) >= 14 else np.nan
    chop_denominator = max_high_14 - min_low_14
    chop_raw = 100 * np.log10(chop_numerator / chop_denominator) / np.log10(14) if chop_denominator > 0 and chop_numerator > 0 else np.nan
    
    # Full choppiness series
    chop_values = []
    for i in range(len(close_1d)):
        if i < 13:
            chop_values.append(np.nan)
            continue
        tr_sum = 0
        for j in range(i-13, i+1):
            tr = max(high_1d[j] - low_1d[j], abs(high_1d[j] - close_1d[j-1] if j>0 else high_1d[j]), abs(low_1d[j] - close_1d[j-1] if j>0 else low_1d[j]))
            tr_sum += tr
        max_h = max(high_1d[i-13:i+1])
        min_l = min(low_1d[i-13:i+1])
        if max_h - min_l > 0:
            chop = 100 * np.log10(tr_sum / (max_h - min_l)) / np.log10(14)
        else:
            chop = np.nan
        chop_values.append(chop)
    chop_values = np.array(chop_values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any data invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Get current 1d volume (aligned)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Regime and confirmation filters
        volume_spike = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        chop_filter = chop_aligned[i] > 61.8  # Range regime
        
        if not (volume_spike and chop_filter):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Camarilla touch conditions
        touch_H3 = high_4h[i] >= H3_aligned[i]  # Touch or penetrate resistance
        touch_L3 = low_4h[i] <= L3_aligned[i]   # Touch or penetrate support
        
        if position == 0:  # Flat - look for mean reversion entries
            # Long at L3 support (price expected to bounce up)
            if touch_L3:
                position = 1
                signals[i] = 0.25
            # Short at H3 resistance (price expected to bounce down)
            elif touch_H3:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Manage position
            # Exit conditions: extreme touch or opposite level touch
            if position == 1:  # Long position
                # Exit if touches H4 (extreme) or touches H3 (opposite resistance)
                if high_4h[i] >= H4_aligned[i] or high_4h[i] >= H3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                # Exit if touches L4 (extreme) or touches L3 (opposite support)
                if low_4h[i] <= L4_aligned[i] or low_4h[i] <= L3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals