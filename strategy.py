#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: On daily timeframe, use Camarilla R1/S1 levels for breakout entries with 1-week EMA34 as trend filter and volume spike confirmation. Designed for low trade frequency (7-25/year) to minimize fee drag. Works in both bull and bear markets by only taking breakouts in direction of weekly trend. Includes close-based stoploss at 2.5x ATR to control drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    # We'll use R1/S1 for breakout and R3/S3 for stronger signals
    R1 = prev_close + 1.125 * prev_range / 12 * 11  # Actually: C + (H-L)*1.125/12
    S1 = prev_close - 1.125 * prev_range / 12 * 11
    R3 = prev_close + 1.125 * prev_range / 6 * 5    # C + (H-L)*1.125/6*5
    S3 = prev_close - 1.125 * prev_range / 6 * 5
    
    # Simpler correct formulas:
    # R4 = C + (H-L)*1.5/2, R3 = C + (H-L)*1.25/2, R2 = C + (H-L)*1.125/2, R1 = C + (H-L)*1.125/12
    # S1 = C - (H-L)*1.125/12, S2 = C - (H-L)*1.125/12, S3 = C - (H-L)*1.25/2, S4 = C - (H-L)*1.5/2
    # Standard Camarilla:
    R1 = prev_close + (prev_high - prev_low) * 1.125 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.125 / 12
    R2 = prev_close + (prev_high - prev_low) * 1.125 / 6
    S2 = prev_close - (prev_high - prev_low) * 1.125 / 6
    R3 = prev_close + (prev_high - prev_low) * 1.125 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.125 / 4
    R4 = prev_close + (prev_high - prev_low) * 1.125 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.125 / 2
    
    # Align 1d Camarilla levels to 1d timeframe (no alignment needed as we're on 1d)
    # But we'll use align_htf_to_ltf for consistency and proper handling
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: current volume > 2.0 * 20-period average (stricter for lower frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 1w EMA34 for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # ATR for stoploss (using 1d data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 20-period vol MA, 34 for EMA, 14 for ATR
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price relative to 1w EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - breakouts of R1/S1 with volume spike and trend alignment
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_spike[i]
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_spike[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.5 * ATR below entry
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at R3 (profit taking)
            elif curr_close >= R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.5 * ATR above entry
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at S3 (profit taking)
            elif curr_close <= S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0