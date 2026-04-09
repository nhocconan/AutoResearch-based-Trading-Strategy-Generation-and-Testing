#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume confirmation + 1w trend filter
# Camarilla levels provide precise intraday support/resistance derived from prior day
# Breakout above H3 or below L3 with volume confirmation captures institutional moves
# 1w EMA filter ensures alignment with higher timeframe trend (avoid counter-trend)
# Works in bull/bear: trend filter adapts, volume confirms authenticity
# Target: 12-37 trades/year (50-150 total over 4 years) with discrete sizing 0.25

name = "12h_1d_1w_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla calculation and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on prior day OHLC)
    # H4, H3, H2, H1, L1, L2, L3, L4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day values (shifted by 1)
    ph = np.concatenate([[np.nan], high_1d[:-1]])
    pl = np.concatenate([[np.nan], low_1d[:-1]])
    pc = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla calculations
    camarilla_h3 = pc + 1.1 * (ph - pl) / 2
    camarilla_l3 = pc - 1.1 * (ph - pl) / 2
    camarilla_h4 = pc + 1.1 * (ph - pl)
    camarilla_l4 = pc - 1.1 * (ph - pl)
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    close_s_1w = pd.Series(close_1w)
    ema_21_1w = close_s_1w.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Align 1w EMA to 12h timeframe (wait for 1w bar close)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Trend filter: price above/below 1w EMA
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR trend shifts to downtrend
            if close[i] < camarilla_l3_aligned[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR trend shifts to uptrend
            if close[i] > camarilla_h3_aligned[i] or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: breakout with volume confirmation in trend direction
            if uptrend and volume_confirmed:
                if close[i] > camarilla_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
            elif downtrend and volume_confirmed:
                if close[i] < camarilla_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals