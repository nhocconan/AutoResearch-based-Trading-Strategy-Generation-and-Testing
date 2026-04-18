#!/usr/bin/env python3
"""
12h Camarilla Pivot Breakout with Volume Confirmation and Daily Trend Filter
Hypothesis: Camarilla pivot levels provide strong support/resistance zones.
In trending markets (price above/below daily EMA34), breaks of R1/S1 levels
with volume confirmation yield high-probability continuation trades.
Uses 1d EMA34 as trend filter to work in both bull and bear markets.
Target: 12-30 trades/year to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop (rule compliance)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = C + ((H-L) * 1.5000), R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666), R1 = C + ((H-L) * 1.0833)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L) * 1.0833), S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500), S4 = C - ((H-L) * 1.5000)
    
    # Use previous day's H, L, C for today's levels
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Calculate pivot levels
    pp = (ph + pl + pc) / 3
    r1 = pc + ((ph - pl) * 1.0833)
    s1 = pc - ((ph - pl) * 1.0833)
    r2 = pc + ((ph - pl) * 1.1666)
    s2 = pc - ((ph - pl) * 1.1666)
    
    # Align to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: price vs daily EMA34
        uptrend = price > ema34_aligned[i]
        downtrend = price < ema34_aligned[i]
        
        # Volume confirmation: above average
        vol_ok = vol > vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume
            if uptrend and vol_ok and price > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend with volume
            elif downtrend and vol_ok and price < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 (reversal) or trend fails
            if price < s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (reversal) or trend fails
            if price > r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_DailyTrend"
timeframe = "12h"
leverage = 1.0