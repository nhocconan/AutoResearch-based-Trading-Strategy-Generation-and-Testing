#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R1 level with 12h EMA50 uptrend and volume > 1.8x 20-bar average
# Short when price breaks below Camarilla S1 level with 12h EMA50 downtrend and volume > 1.8x 20-bar average
# Exit via ATR(14) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Camarilla pivot levels derived from previous 12h bar's high/low/close for proper alignment
# Target: 75-200 total trades over 4 years = 19-50/year. Uses discrete sizing (0.30) to balance return and fee drag.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
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
    
    # Get 12h data ONCE before loop (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter (aligned to 4h)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for calculations)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from PREVIOUS completed 12h bar
        # Need at least one completed 12h bar to calculate levels
        if len(df_12h) == 0:
            signals[i] = 0.0
            continue
            
        # Find the index of the last completed 12h bar
        # Use the aligned timestamp approach: get the 12h bar that closed before current 4h bar
        current_time = prices.iloc[i]['open_time']
        # Find 12h bars that closed before or at current time
        mask = df_12h['open_time'] <= current_time
        if not mask.any():
            signals[i] = 0.0
            continue
            
        last_12h_idx = mask.sum() - 1  # Index of last completed 12h bar
        if last_12h_idx < 0:
            signals[i] = 0.0
            continue
            
        # Get high, low, close of the last completed 12h bar
        prev_high = df_12h.iloc[last_12h_idx]['high']
        prev_low = df_12h.iloc[last_12h_idx]['low']
        prev_close = df_12h.iloc[last_12h_idx]['close']
        
        # Calculate Camarilla R1 and S1 levels
        camarilla_range = prev_high - prev_low
        r1 = prev_close + camarilla_range * 1.1 / 12
        s1 = prev_close - camarilla_range * 1.1 / 12
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R1 with 12h EMA50 uptrend and volume spike
            if close[i] > r1 and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below Camarilla S1 with 12h EMA50 downtrend and volume spike
            elif close[i] < s1 and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.5 * ATR
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.5 * ATR
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals