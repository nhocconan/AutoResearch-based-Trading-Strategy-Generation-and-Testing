#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x average)
    # Camarilla pivot levels from 1d provide high-probability reversal/continuation points from daily structure
    # 1d EMA34 filters for longer-term trend alignment to avoid counter-trend whipsaws
    # Volume spike >1.5x 20-period average confirms institutional participation
    # Exits on H3/L3 retest or trend reversal
    # Target: 12-37 trades/year (50-150 total over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate previous 12h bar's Camarilla levels (H3, L3)
    # Based on previous 12h bar's range
    camarilla_h3 = np.full(len(high_12h), np.nan)
    camarilla_l3 = np.full(len(low_12h), np.nan)
    
    for i in range(1, len(high_12h)):
        # Use previous bar's high/low/close for Camarilla calculation
        ph = high_12h[i-1]
        pl = low_12h[i-1]
        pc = close_12h[i-1]
        rang = ph - pl
        
        camarilla_h3[i] = pc + rang * 1.1 / 4  # H3 level
        camarilla_l3[i] = pc - rang * 1.1 / 4  # L3 level
    
    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_12h[i] = np.mean(volume[i-20:i])
    volume_spike_12h = volume > (1.5 * vol_ma_12h)
    
    # Align all indicators to LTF (12h)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_12h_aligned[i]) or np.isnan(l3_12h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike_12h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > h3_12h_aligned[i]
        short_breakout = close[i] < l3_12h_aligned[i]
        
        # 1d trend filter (EMA34)
        bullish_trend = close[i] > ema34_1d_aligned[i]
        bearish_trend = close[i] < ema34_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_12h[i]
        short_entry = short_breakout and bearish_trend and volume_spike_12h[i]
        
        # Exit logic: price retests H3/L3 or trend reversal
        long_exit = (close[i] <= h3_12h_aligned[i] * 1.001) or not bullish_trend  # Retest H3 or trend change
        short_exit = (close[i] >= l3_12h_aligned[i] * 0.999) or not bearish_trend  # Retest L3 or trend change
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_h3l3_ema34_volume_v1"
timeframe = "12h"
leverage = 1.0