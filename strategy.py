#!/usr/bin/env python3
"""
6h_RelativeStrength_Momentum
Hypothesis: Use relative strength (RS) vs Bitcoin to capture momentum leadership in altcoins.
Long when ETH/SOL shows RS strength (price outperforming BTC) with 60-period price above 200EMA and rising RSI.
Short when RS weakness (underperforming BTC) with price below 200EMA and falling RSI.
Works in bull/bear by following leadership shifts; RS filters out weak moves.
Target: 15-25 trades/year (60-100 total) to minimize fee drag.
"""

name = "6h_RelativeStrength_Momentum"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly BTC data for RS calculation
    df_1w_btc = get_htf_data(prices, '1w')
    btc_close = df_1w_btc['close'].values
    
    # Calculate 60-period price change (momentum)
    price_change_60 = np.full(n, np.nan)
    for i in range(60, n):
        price_change_60[i] = (close[i] - close[i-60]) / close[i-60]
    
    # Calculate weekly BTC momentum for comparison
    btc_change_60 = np.full(len(btc_close), np.nan)
    for i in range(60, len(btc_close)):
        btc_change_60[i] = (btc_close[i] - btc_close[i-60]) / btc_close[i-60]
    
    # Align BTC momentum to 6h
    btc_change_60_aligned = align_htf_to_ltf(prices, df_1w_btc, btc_change_60)
    
    # Relative Strength: asset momentum minus BTC momentum
    rs_momentum = price_change_60 - btc_change_60_aligned
    
    # 200-period EMA for trend filter
    ema200 = np.full(n, np.nan)
    if n >= 200:
        ema200[199] = np.mean(close[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, n):
            ema200[i] = alpha * close[i] + (1 - alpha) * ema200[i-1]
    
    # 14-period RSI for momentum confirmation
    rsi = np.full(n, np.nan)
    if n >= 14:
        # Calculate first average gain/loss
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
            rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
            rsi[i] = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(rs_momentum[i]) or np.isnan(ema200[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Conditions for long: positive RS momentum, price above EMA200, RSI rising
        long_condition = (rs_momentum[i] > 0 and 
                         close[i] > ema200[i] and 
                         rsi[i] > rsi[i-1])
        
        # Conditions for short: negative RS momentum, price below EMA200, RSI falling
        short_condition = (rs_momentum[i] < 0 and 
                          close[i] < ema200[i] and 
                          rsi[i] < rsi[i-1])
        
        if position == 0:
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RS turns negative or price falls below EMA200
            if rs_momentum[i] <= 0 or close[i] <= ema200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RS turns positive or price rises above EMA200
            if rs_momentum[i] >= 0 or close[i] >= ema200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals