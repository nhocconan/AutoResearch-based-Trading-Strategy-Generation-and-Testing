#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R1 level with 12h EMA50 uptrend and volume > 1.5x 20-bar average
# Short when price breaks below Camarilla S1 level with 12h EMA50 downtrend and volume > 1.5x 20-bar average
# Exit via ATR(14) trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR
#                      short exit when price > lowest_low_since_entry + 2.0 * ATR
# Uses Camarilla pivot levels from 12h timeframe for structure, 12h EMA50 for trend filter, volume for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 75-200 total trades over 4 years = 19-50/year.

name = "4h_Camarilla_R1S1_12hEMA50_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla pivot levels (using prior 12h bar's OHLC)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Prior 12h bar's OHLC for Camarilla calculation
    ph = df_12h['high'].shift(1).values  # prior 12h high
    pl = df_12h['low'].shift(1).values   # prior 12h low
    pc = df_12h['close'].shift(1).values # prior 12h close
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = pc + (ph - pl) * 1.1 / 12
    camarilla_s1 = pc - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (completed 12h bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for EMA50 and ATR calculations)
    start_idx = 50  # EMA50 needs 50 bars, plus buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R1 with 12h EMA50 uptrend and volume spike
            if close[i] > camarilla_r1_aligned[i] and ema_50_aligned[i] > ema_50_aligned[i-1] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below Camarilla S1 with 12h EMA50 downtrend and volume spike
            elif close[i] < camarilla_s1_aligned[i] and ema_50_aligned[i] < ema_50_aligned[i-1] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.0 * ATR
            if close[i] < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.0 * ATR
            if close[i] > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals