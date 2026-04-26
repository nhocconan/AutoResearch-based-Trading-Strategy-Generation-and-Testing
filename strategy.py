#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_WeeklyTrend_ATR_Exit_v1
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA50 trend filter and volume spike, exited by ATR-based trailing stop. Designed for 1d timeframe to target 7-25 trades/year with discrete sizing (0.25). Works in bull/bear via weekly trend alignment and volatility-based exits.
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Previous day's Camarilla levels (using 1d OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla calculations
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align to 1d timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    
    # Warmup: max of weekly EMA50 (50), volume MA (20), ATR (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_50_1w_aligned[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        r3_val = R3_aligned[i]
        s3_val = S3_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price vs weekly EMA50
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: price breaks above R1 with weekly uptrend and volume spike
        long_condition = (close_val > r1_val) and uptrend and vol_spike
        # Short: price breaks below S1 with weekly downtrend and volume spike
        short_condition = (close_val < s1_val) and downtrend and vol_spike
        
        # Exit conditions
        long_exit = False
        short_exit = False
        
        if position == 1:
            # Trailing stop: exit if price drops below highest high since entry minus 2.5*ATR
            if entry_price > 0:
                highest_since_entry = np.max(high[entry_idx:i+1]) if 'entry_idx' in locals() else high_val
                long_exit = low_val < (highest_since_entry - 2.5 * atr_val)
            # Also exit if price re-enters R3-S3 range (mean reversion in strong trends)
            if not long_exit and (r3_val <= close_val <= s3_val):
                long_exit = True
        elif position == -1:
            # Trailing stop: exit if price rises above lowest low since entry plus 2.5*ATR
            if entry_price > 0:
                lowest_since_entry = np.min(low[entry_idx:i+1]) if 'entry_idx' in locals() else low_val
                short_exit = high_val > (lowest_since_entry + 2.5 * atr_val)
            # Also exit if price re-enters R3-S3 range
            if not short_exit and (r3_val <= close_val <= s3_val):
                short_exit = True
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
            entry_idx = i
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
            entry_idx = i
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_WeeklyTrend_ATR_Exit_v1"
timeframe = "1d"
leverage = 1.0