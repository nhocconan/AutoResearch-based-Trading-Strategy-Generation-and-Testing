#!/usr/bin/env python3
"""
12h_1w_1d_Momentum_Trap_V1
Hypothesis: Combines weekly trend filter (price above/below weekly EMA20) with daily momentum exhaustion signals.
Long when weekly uptrend AND daily RSI<30 AND price>daily VWAP; short when weekly downtrend AND daily RSI>70 AND price<daily VWAP.
Designed for low trade frequency by requiring alignment of weekly trend and daily oversold/overbought conditions.
Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Momentum_Trap_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_ws = pd.Series(close_1w)
    ema20_w = close_ws.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_w_aligned = align_htf_to_ltf(prices, df_1w, ema20_w)
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily RSI(14)
    close_ds = pd.Series(close_1d)
    delta = close_ds.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Daily VWAP (typical price * volume)
    typical_price = (high_1d + low_1d + close_1d) / 3
    vp = typical_price * volume_1d
    cum_vp = np.nancumsum(vp)
    cum_vol = np.nancumsum(volume_1d)
    vwap = cum_vp / cum_vol
    vwap_values = vwap
    
    # Align daily indicators
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(ema20_w_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vwap_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema20_w_aligned[i]
        weekly_downtrend = close[i] < ema20_w_aligned[i]
        
        # Daily momentum exhaustion
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        price_above_vwap = close[i] > vwap_aligned[i]
        price_below_vwap = close[i] < vwap_aligned[i]
        
        # Entry conditions
        long_setup = weekly_uptrend and rsi_oversold and price_above_vwap
        short_setup = weekly_downtrend and rsi_overbought and price_below_vwap
        
        # Exit when conditions reverse
        exit_long = not (weekly_uptrend and rsi_oversold and price_above_vwap)
        exit_short = not (weekly_downtrend and rsi_overbought and price_below_vwap)
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals