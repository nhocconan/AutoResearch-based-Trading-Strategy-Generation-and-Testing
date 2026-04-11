#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v27
# Strategy: 4h breakout of Camarilla pivot levels from 1d with volume confirmation and RSI filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels act as key support/resistance on daily timeframe.
# Breakouts above H4 or below L4 with volume confirmation signal institutional participation.
# RSI filter avoids counter-trend entries. Designed for low trade frequency (~20-40/year).
# Works in bull markets via breakout continuation and bear markets via breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v27"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L), H3 = C + 1.1*(H-L), L3 = C - 1.1*(H-L)
    # H2 = C + 0.5*(H-L), L2 = C - 0.5*(H-L), H1 = C + 0.25*(H-L), L1 = C - 0.25*(H-L)
    # Pivot = (H+L+C)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first day's values to zero (will be handled by min_periods later)
    prev_high[0] = 0
    prev_low[0] = 0
    prev_close[0] = 0
    
    # Calculate Camarilla levels for previous day
    rang = prev_high - prev_low
    H4 = prev_close + 1.5 * rang
    L4 = prev_close - 1.5 * rang
    H3 = prev_close + 1.1 * rang
    L3 = prev_close - 1.1 * rang
    
    # Align to 4h timeframe (these levels are valid for the entire day after the 1d bar closes)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current 1d volume > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # RSI filter on 4h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(H3_aligned[i]) or \
           np.isnan(L3_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]) or \
           np.isnan(rsi[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # RSI filter: >50 for bullish bias, <50 for bearish bias
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Entry conditions
        # Long: Break above H4 with volume confirmation and bullish RSI
        if close[i] > H4_aligned[i] and vol_confirm and rsi_bullish and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Break below L4 with volume confirmation and bearish RSI
        elif close[i] < L4_aligned[i] and vol_confirm and rsi_bearish and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price returns to mean reversion zone (between H3 and L3)
        elif position == 1 and close[i] < H3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > L3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals