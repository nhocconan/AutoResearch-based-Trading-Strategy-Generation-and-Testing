#!/usr/bin/env python3
"""
1h_4d_Pullback_to_VWAP_with_Momentum_Filter
Hypothesis: In strong trends (4h EMA50 > EMA200), price pulls back to 1h VWAP offering high-probability entry.
Momentum filter (1h RSI > 50 for longs, < 50 for shorts) ensures alignment with short-term momentum.
Works in bull (buy pullbacks) and bear (sell rallies) by trading with the 4h trend.
Targets 20-40 trades/year via strict trend + pullback + momentum confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA50 > EMA200 for uptrend, < for downtrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # 1h VWAP calculation (typical price * volume)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_tpv, cum_vol, out=np.full_like(cum_tpv, np.nan), where=cum_vol!=0)
    
    # 1h RSI for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vwap[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 4h
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Long: uptrend + price at or below VWAP + RSI > 50 (bullish momentum)
        long_condition = uptrend and (close[i] <= vwap[i]) and (rsi[i] > 50)
        
        # Short: downtrend + price at or above VWAP + RSI < 50 (bearish momentum)
        short_condition = downtrend and (close[i] >= vwap[i]) and (rsi[i] < 50)
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1h_4d_Pullback_to_VWAP_with_Momentum_Filter"
timeframe = "1h"
leverage = 1.0