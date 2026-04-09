#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot long/short + 1d volume spike + chop regime filter
# Camarilla levels provide institutional support/resistance; volume confirms breakout authenticity
# Chop filter avoids whipsaws in ranging markets; discrete sizing 0.25 limits drawdown
# Works in bull/bear: Camarilla levels adapt to volatility, volume works in both regimes
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing

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
    
    # Load 1d data ONCE before loop for Camarilla and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    camarilla_h4 = np.full(n, np.nan)  # Long entry level
    camarilla_l4 = np.full(n, np.nan)  # Short entry level
    camarilla_h3 = np.full(n, np.nan)  # Long stop level
    camarilla_l3 = np.full(n, np.nan)  # Short stop level
    
    for i in range(n):
        if i < 1:  # Need previous day's data
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
        else:
            # Get previous day's OHLC (1d bar at i-1)
            prev_high = df_1d['high'].iloc[i-1] if i-1 < len(df_1d) else np.nan
            prev_low = df_1d['low'].iloc[i-1] if i-1 < len(df_1d) else np.nan
            prev_close = df_1d['close'].iloc[i-1] if i-1 < len(df_1d) else np.nan
            
            if np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close):
                camarilla_h4[i] = np.nan
                camarilla_l4[i] = np.nan
                camarilla_h3[i] = np.nan
                camarilla_l3[i] = np.nan
            else:
                # Camarilla formulas
                range_val = prev_high - prev_low
                camarilla_h4[i] = prev_close + range_val * 1.1 / 2
                camarilla_l4[i] = prev_close - range_val * 1.1 / 2
                camarilla_h3[i] = prev_close + range_val * 1.1 / 4
                camarilla_l3[i] = prev_close - range_val * 1.1 / 4
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    chop = np.full(n, np.nan)
    if len(df_1d) >= 14:
        # True Range
        tr = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1))
        tr = np.maximum(tr, np.roll(df_1d['low'].values, 1))
        tr = np.maximum(tr - np.roll(tr, 1), np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)))
        tr = np.maximum(tr, np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1)))
        tr[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]  # First bar
        
        # Smoothing
        atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        
        # Chop calculation
        for i in range(n):
            idx_1d = i // 96  # Approximate: 96 4h bars per 1d (24*60/15)
            if idx_1d < 14 or idx_1d >= len(df_1d):
                chop[i] = np.nan
            else:
                # Sum of true range over last 14 periods
                tr_sum = np.sum(tr[max(0, idx_1d-13):idx_1d+1])
                # Highest high and lowest low over last 14 periods
                hh = np.max(df_1d['high'].iloc[max(0, idx_1d-13):idx_1d+1].values)
                ll = np.min(df_1d['low'].iloc[max(0, idx_1d-13):idx_1d+1].values)
                if hh == ll:
                    chop[i] = 50.0  # Avoid division by zero
                else:
                    chop[i] = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    
    # Align 1d indicators to 4h timeframe (wait for 1d bar close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 20-period average volume for volume confirmation
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
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * avg_volume[i]
        
        # Chop regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
        # We use CHOP > 50 for mean reversion tendency (avoid strong trends)
        chop_filter = chop_aligned[i] > 50
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 (stop) OR price > Camarilla H4 (profit target)
            if close[i] < camarilla_l3_aligned[i] or close[i] > camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 (stop) OR price < Camarilla L4 (profit target)
            if close[i] > camarilla_h3_aligned[i] or close[i] < camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, chop filter, and Camarilla levels
            if volume_confirmed and chop_filter:
                # Long entry: price > Camarilla H4 AND price < Camarilla H3 (fade false breakout)
                if close[i] > camarilla_h4_aligned[i] and close[i] < camarilla_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Camarilla L4 AND price > Camarilla L3 (fade false breakout)
                elif close[i] < camarilla_l4_aligned[i] and close[i] > camarilla_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals