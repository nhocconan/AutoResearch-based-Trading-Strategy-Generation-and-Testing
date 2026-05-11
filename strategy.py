#!/usr/bin/env python3
name = "6h_1d_1w_Hybrid_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly momentum: price above weekly EMA20 and rising
    weekly_ema20 = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_prev = np.roll(weekly_ema20, 1)
    weekly_ema20_prev[0] = np.nan
    weekly_uptrend = (df_1w['close'].values > weekly_ema20) & (weekly_ema20 > weekly_ema20_prev)
    
    # Daily momentum: RSI(14) > 50 and rising
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_rising = rsi_1d > rsi_1d.shift(1)
    daily_momentum = (rsi_1d > 50) & rsi_rising
    
    # 6h price momentum: close > open and volume confirmation
    price_up = close > prices['open'].values
    vol_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_confirm = volume > 1.2 * vol_ma10
    
    # Align HTF momentum to 6h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    daily_momentum_aligned = align_htf_to_ltf(prices, df_1d, daily_momentum)
    
    # Entry conditions: weekly uptrend + daily momentum + 6s price/volume confirmation
    long_signal = weekly_uptrend_aligned & daily_momentum_aligned & price_up & volume_confirm
    
    # Weekly momentum: price below weekly EMA20 and falling
    weekly_downtrend = (df_1w['close'].values < weekly_ema20) & (weekly_ema20 < weekly_ema20_prev)
    
    # Daily momentum: RSI(14) < 50 and falling
    daily_weak = (rsi_1d < 50) & (~rsi_rising)
    
    # 6h price momentum: close < open and volume confirmation
    price_down = close < prices['open'].values
    
    # Align HTF weakness to 6h
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    daily_weak_aligned = align_htf_to_ltf(prices, df_1d, daily_weak)
    
    # Exit/short conditions: weekly downtrend + daily weakness + 6s price/volume confirmation
    short_signal = weekly_downtrend_aligned & daily_weak_aligned & price_down & volume_confirm
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(weekly_uptrend_aligned[i]) or np.isnan(daily_momentum_aligned[i]) or
            np.isnan(weekly_downtrend_aligned[i]) or np.isnan(daily_weak_aligned[i]) or
            np.isnan(vol_ma10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            if long_signal[i]:
                signals[i] = 0.25
                position = 1
            elif short_signal[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly uptrend breaks or daily momentum fades
            if not (weekly_uptrend_aligned[i] and daily_momentum_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly downtrend breaks or daily weakness fades
            if not (weekly_downtrend_aligned[i] and daily_weak_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals