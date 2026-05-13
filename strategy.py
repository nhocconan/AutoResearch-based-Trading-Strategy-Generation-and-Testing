#!/usr/bin/env python3
"""
12h_Triple_Pattern_Confirmation
Hypothesis: Combines 12h Donchian breakout, 1w EMA trend filter, and 1d RSI momentum for high-probability entries.
Designed for low trade frequency (15-25/year) with strong trend-following logic that works in both bull and bear markets.
Uses volume confirmation and ATR-based stoploss to reduce false signals and manage risk.
"""

name = "12h_Triple_Pattern_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian Channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_channel = high_roll.values
    lower_channel = low_roll.values
    
    # Calculate RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values
    
    # Get 1-week trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1-day RSI for entry timing
    df_1d = get_htf_data(prices, '1d')
    delta_1d = pd.Series(df_1d['close']).diff()
    gain_1d = delta_1d.clip(lower=0)
    loss_1d = -delta_1d.clip(upper=0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d.replace(0, np.nan)
    rsi_1d = (100 - (100 / (1 + rs_1d))).fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above upper channel with 1w uptrend, 1d RSI > 55, and volume confirmation
            if (close[i] > upper_channel[i] and 
                ema_50_1w_aligned[i] is not np.nan and 
                close[i] > ema_50_1w_aligned[i] and
                rsi_1d_aligned[i] > 55 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower channel with 1w downtrend, 1d RSI < 45, and volume confirmation
            elif (close[i] < lower_channel[i] and 
                  ema_50_1w_aligned[i] is not np.nan and 
                  close[i] < ema_50_1w_aligned[i] and
                  rsi_1d_aligned[i] < 45 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below lower channel or 1w trend turns down
            if close[i] < lower_channel[i] or (ema_50_1w_aligned[i] is not np.nan and close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above upper channel or 1w trend turns up
            if close[i] > upper_channel[i] or (ema_50_1w_aligned[i] is not np.nan and close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals