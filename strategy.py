#!/usr/bin/env python3
# 6h_12h_1d_rsi_divergence_volume_v1
# Strategy: 6s RSI divergence with 12h trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: RSI divergence identifies exhaustion in trends. Combined with 12h EMA trend filter and volume spike,
# it provides high-probability reversal entries. Works in both bull (buy dips) and bear (sell rallies) markets.
# Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_rsi_divergence_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 30 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d RSI(14) for divergence detection
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # 6h RSI(14) for entry confirmation
    delta_6h = pd.Series(close).diff()
    gain_6h = delta_6h.where(delta_6h > 0, 0)
    loss_6h = -delta_6h.where(delta_6h < 0, 0)
    avg_gain_6h = gain_6h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_6h = loss_6h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_6h = avg_gain_6h / avg_loss_6h
    rsi_6h = 100 - (100 / (1 + rs_6h))
    rsi_6h_values = rsi_6h.values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(rsi_6h_values[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # RSI values
        rsi_1d_now = rsi_1d_aligned[i]
        rsi_1d_prev = rsi_1d_aligned[i-1] if i > 0 else 50
        rsi_6h_now = rsi_6h_values[i]
        rsi_6h_prev = rsi_6h_values[i-1] if i > 0 else 50
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i >= 2:
            # Check for lower low in price and higher low in RSI over last 2 periods
            if close[i] < close[i-2] and low[i] < low[i-2]:
                if rsi_1d_now > rsi_1d_prev and rsi_6h_now > rsi_6h_prev:
                    bullish_div = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i >= 2:
            # Check for higher high in price and lower high in RSI over last 2 periods
            if close[i] > close[i-2] and high[i] > high[i-2]:
                if rsi_1d_now < rsi_1d_prev and rsi_6h_now < rsi_6h_prev:
                    bearish_div = True
        
        # Entry logic: RSI divergence + volume spike + trend alignment (counter-trend)
        if bullish_div and volume_spike[i] and downtrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_div and volume_spike[i] and uptrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: trend resumes or RSI reaches extreme
        elif position == 1 and (rsi_1d_now > 70 or uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_1d_now < 30 or downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals