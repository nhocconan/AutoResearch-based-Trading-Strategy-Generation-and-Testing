#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot levels with 12h volume confirmation and 1d trend filter
    # Camarilla provides intraday support/resistance levels based on previous period
    # Long at S1, short at R1 with volume confirmation; filter by 1d EMA50 trend
    # This provides mean reversion in range and continuation in trend, working in both bull/bear
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Camarilla calculation (previous period)
    df_12h = get_htf_data(prices, '12h')
    # Calculate Camarilla levels from previous 12h bar
    if len(df_12h) < 2:
        return np.zeros(n)
    ph = df_12h['high'].iloc[-2]  # previous 12h high
    pl = df_12h['low'].iloc[-2]   # previous 12h low
    pc = df_12h['close'].iloc[-2] # previous 12h close
    # Camarilla levels
    r4 = pc + ((ph - pl) * 1.5000)
    r3 = pc + ((ph - pl) * 1.2500)
    r2 = pc + ((ph - pl) * 1.1666)
    r1 = pc + ((ph - pl) * 1.0833)
    s1 = pc - ((ph - pl) * 1.0833)
    s2 = pc - ((ph - pl) * 1.1666)
    s3 = pc - ((ph - pl) * 1.2500)
    s4 = pc - ((ph - pl) * 1.5000)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation (24-period MA on 6h)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma24  # Above average volume
    
    signals = np.zeros(n)
    
    for i in range(24, n):  # Start after volume MA warmup
        # Skip if trend data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma24[i]):
            continue
            
        # Only trade when we have valid Camarilla levels (from previous 12h bar)
        # The levels are constant within the current 12h period
        if i < 24:  # Need at least one full 12h bar before
            continue
            
        # Long: price at or below S1 with volume confirmation and above 1d EMA50 (uptrend filter)
        if close[i] <= s1 and vol_confirm[i] and close[i] > ema50_1d_aligned[i]:
            signals[i] = 0.25
        # Short: price at or above R1 with volume confirmation and below 1d EMA50 (downtrend filter)
        elif close[i] >= r1 and vol_confirm[i] and close[i] < ema50_1d_aligned[i]:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_S1_R1_12hVol_1dEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0