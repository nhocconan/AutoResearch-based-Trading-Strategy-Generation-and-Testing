#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout for direction and 1h RSI for entry timing.
# 4h Donchian captures major trend direction, 1h RSI provides pullback entries in trending markets.
# This reduces whipsaw and avoids counter-trend trading.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Session filter (08-20 UTC) reduces noise trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    period = 20
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = np.full(len(df_4h), np.nan)
    donchian_low_4h = np.full(len(df_4h), np.nan)
    for i in range(period-1, len(df_4h)):
        donchian_high_4h[i] = np.max(high_4h[i-period+1:i+1])
        donchian_low_4h[i] = np.min(low_4h[i-period+1:i+1])
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # 1h RSI (14-period) for entry timing
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(rsi_period, n):
        if i == rsi_period:
            avg_gain[i] = np.mean(gain[i-rsi_period:i])
            avg_loss[i] = np.mean(loss[i-rsi_period:i])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        price = close[i]
        donch_high = donchian_high_4h_aligned[i]
        donch_low = donchian_low_4h_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: 4h uptrend (price above 4h Donchian mid) + RSI pullback (< 40)
            if (price > (donch_high + donch_low) / 2 and 
                rsi_val < 40):
                position = 1
                signals[i] = position_size
            # Short: 4h downtrend (price below 4h Donchian mid) + RSI bounce (> 60)
            elif (price < (donch_high + donch_low) / 2 and 
                  rsi_val > 60):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: 4h trend reversal (price below 4h Donchian low) OR RSI overbought
            if (price < donch_low or 
                rsi_val > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: 4h trend reversal (price above 4h Donchian high) OR RSI oversold
            if (price > donch_high or 
                rsi_val < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_Donchian_RSI_Pullback_v1"
timeframe = "1h"
leverage = 1.0