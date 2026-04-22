# 6h_WeeklyPivot_R2_S2_Breakout_1dEMA50_Volume
# Hypothesis: Weekly pivot levels (R2/S2) provide strong support/resistance. Breakouts above R2 or below S2 with volume confirmation and alignment to daily EMA50 trend yield high-probability trades. Weekly pivots filter noise, daily EMA50 ensures trend alignment, volume confirms breakout strength. Works in bull/bear by following trend.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Load weekly data once for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (R2, S2)
    pp_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r2_1w = pp_1w + range_1w  # R2 = PP + (High - Low)
    s2_1w = pp_1w - range_1w  # S2 = PP - (High - Low)
    
    # Align weekly pivots to 6h
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Volume spike filter (24-period average on 6h)
    volume = prices['volume'].values
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 24-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above R2 + volume spike + price > EMA50
            if price > r2 and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S2 + volume spike + price < EMA50
            elif price < s2 and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through weekly PP or volume dries up
            pp_1w_today = (high_1w[i // 28] + low_1w[i // 28] + close_1w[i // 28]) / 3 if i >= 28 else pp_1w[0]
            pp_aligned_today = align_htf_to_ltf(prices, df_1w, np.full_like(pp_1w, pp_1w_today))[i] if i >= 28 else pp_1w[0]
            
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below weekly PP or volume dries up
                if price < pp_aligned_today or vol < 0.5 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above weekly PP or volume dries up
                if price > pp_aligned_today or vol < 0.5 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_R2_S2_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0