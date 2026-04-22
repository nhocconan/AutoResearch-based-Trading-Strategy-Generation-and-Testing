#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h momentum with 4h trend filter and volume confirmation
    # Uses RSI(14) for momentum, 4h EMA(50) for trend direction, and volume spike for confirmation
    # Designed to capture momentum bursts in both bull and bear markets with controlled trade frequency
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma20  # Require 1.8x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after RSI warmup
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI > 55 (bullish momentum) + price above 4h EMA50 + volume spike
            if rsi[i] > 55 and close[i] > ema50_4h_aligned[i] and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI < 45 (bearish momentum) + price below 4h EMA50 + volume spike
            elif rsi[i] < 45 and close[i] < ema50_4h_aligned[i] and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI crosses 50 (momentum fade) or price crosses 4h EMA50 (trend change)
            if position == 1:
                if rsi[i] < 50 or close[i] < ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi[i] > 50 or close[i] > ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI_Momentum_4hEMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0