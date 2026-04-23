#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1-week Camarilla pivot levels (R4/S4) for breakout direction and 1-day EMA34 for trend filter.
Long when price breaks above weekly R4 AND close > 1d EMA34 (bullish alignment).
Short when price breaks below weekly S4 AND close < 1d EMA34 (bearish alignment).
Exit when price retraces to weekly midpoint (R3/S3 level) or ATR trailing stop (2.0*ATR) triggered.
Uses volume confirmation (volume > 1.3x 20-period average) to filter false breakouts.
Designed for 6h timeframe targeting ~15-25 trades/year per symbol (60-100 total over 4 years).
Focus on BTC and ETH as primary targets with multi-timeframe alignment to avoid whipsaws.
Weekly Camarilla provides strong structural levels; 1d EMA34 filters counter-trend noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly OHLC for Camarilla calculation
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Camarilla levels: R4 = C + ((H-L)*1.1/2), S4 = C - ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    # Midpoint = (R3 + S3)/2 = C (weekly close)
    diff = h_1w - l_1w
    r4 = c_1w + (diff * 1.1 / 2)
    s4 = c_1w - (diff * 1.1 / 2)
    r3 = c_1w + (diff * 1.1 / 4)
    s3 = c_1w - (diff * 1.1 / 4)
    midpoint = c_1w  # Weekly close as pivot midpoint
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    midpoint_aligned = align_htf_to_ltf(prices, df_1w, midpoint)
    
    # Calculate 1-day EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Vol MA needs 20, EMA needs 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        midpoint_val = midpoint_aligned[i]
        ema_34_val = ema_34_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly R4 AND volume spike AND bullish trend (close > EMA34)
            if price > r4_val and volume[i] > 1.3 * vol_ma_val and close[i] > ema_34_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Price breaks below weekly S4 AND volume spike AND bearish trend (close < EMA34)
            elif price < s4_val and volume[i] > 1.3 * vol_ma_val and close[i] < ema_34_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to weekly midpoint (R3/S3 level)
            if position == 1 and price <= midpoint_val:
                exit_signal = True
            elif position == -1 and price >= midpoint_val:
                exit_signal = True
            
            # Secondary exit: Price reaches R3/S3 (take profit zone)
            if position == 1 and price >= r3_val:
                exit_signal = True
            elif position == -1 and price <= s3_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WeeklyCamarilla_R4S4_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0