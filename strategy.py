#!/usr/bin/env python3
"""
4h_RSI20_Pullback_1dTrend_VolumeFilter
Hypothesis: Enter long when RSI(2) < 20 (deep oversold) and price above daily EMA34 with volume confirmation; enter short when RSI(2) > 80 and price below daily EMA34 with volume confirmation. Exit when RSI(2) crosses above 80 (for longs) or below 20 (for shorts). Uses daily trend filter to avoid counter-trend trades. Designed for 4h timeframe to target 20-40 trades/year, avoiding fee drag while capturing mean-reversion moves in both bull and bear markets.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # RSI(2) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[1] = gain[1]
    avg_loss[1] = loss[1]
    for i in range(2, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI and EMA
    start_idx = max(34, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        rsi_val = rsi[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: RSI < 20 (oversold) + volume spike + uptrend (price > EMA34)
            if rsi_val < 20 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: RSI > 80 (overbought) + volume spike + downtrend (price < EMA34)
            elif rsi_val > 80 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 80 (overbought) or trend turns down
            if rsi_val > 80 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI crosses below 20 (oversold) or trend turns up
            if rsi_val < 20 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RSI20_Pullback_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0