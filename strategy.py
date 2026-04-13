#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x average)
    # 12h timeframe reduces trade frequency to avoid fee drag while capturing multi-day swings
    # Camarilla H4/L4 levels from 1d provide significant intraday support/resistance
    # 1d EMA34 filter ensures alignment with daily trend to avoid counter-trend whipsaws
    # Volume spike >1.5x 20-period average confirms participation
    # Target: 12-30 trades/year (50-120 total over 4 years) for minimal fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous 1d bar's Camarilla levels (H4, L4)
    camarilla_h4 = np.full(len(high_1d), np.nan)
    camarilla_l4 = np.full(len(low_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        # Use previous bar's high/low/close for Camarilla calculation
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        rang = ph - pl
        
        camarilla_h4[i] = pc + rang * 1.1 / 2  # H4 level
        camarilla_l4[i] = pc - rang * 1.1 / 2  # L4 level
    
    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_12h[i] = np.mean(volume[i-20:i])
    volume_spike_12h = volume > (1.5 * vol_ma_12h)
    
    # Align all indicators to LTF (12h)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike_12h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > h4_1d_aligned[i]
        short_breakout = close[i] < l4_1d_aligned[i]
        
        # 1d trend filter (EMA34)
        bullish_trend = close[i] > ema34_1d_aligned[i]
        bearish_trend = close[i] < ema34_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_12h[i]
        short_entry = short_breakout and bearish_trend and volume_spike_12h[i]
        
        # Exit logic: price retests H4/L4 or trend reversal
        long_exit = (close[i] <= h4_1d_aligned[i] * 1.001) or not bullish_trend  # Retest H4 or trend change
        short_exit = (close[i] >= l4_1d_aligned[i] * 0.999) or not bearish_trend  # Retest L4 or trend change
        
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

name = "12h_1d_camarilla_h4l4_ema34_volume_v1"
timeframe = "12h"
leverage = 1.0