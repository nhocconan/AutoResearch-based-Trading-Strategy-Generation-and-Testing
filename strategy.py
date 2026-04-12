#!/usr/bin/env python3
"""
4h_1d_RSI_Reversion_V1
Hypothesis: RSI mean reversion with 1d trend filter and volume confirmation.
Long when RSI(14) < 30 and price > 1d EMA50; short when RSI(14) > 70 and price < 1d EMA50.
Designed for low trade frequency by requiring extreme RSI + trend alignment + volume.
Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RSI_Reversion_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA50 for trend filter
    close_s = pd.Series(close_1d)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    gain_sum = 0.0
    loss_sum = 0.0
    for i in range(n):
        gain_sum += gain[i]
        loss_sum += loss[i]
        if i >= 14:
            gain_sum -= gain[i-14]
            loss_sum -= loss[i-14]
        if i >= 13:
            avg_gain[i] = gain_sum / 14
            avg_loss[i] = loss_sum / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period for confirmation)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    # Align daily EMA50 to 4h
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.2x average
        vol_confirm = volume[i] > 1.2 * vol_avg[i]
        
        # Trend filter: price above/below daily EMA50
        price_vs_ema = close[i] > ema50_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Entry conditions
        long_setup = rsi_oversold and price_vs_ema and vol_confirm
        short_setup = rsi_overbought and not price_vs_ema and vol_confirm
        
        # Exit when RSI returns to neutral or trend fails
        exit_long = rsi[i] >= 50 or not price_vs_ema
        exit_short = rsi[i] <= 50 or price_vs_ema
        
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