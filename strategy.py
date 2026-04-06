#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy with 4h trend filter and volume confirmation.
# Uses 4h RSI for trend direction (RSI > 50 = uptrend, < 50 = downtrend).
# 1h RSI for entry timing (oversold/overbought within trend).
# Volume filter: current volume > 1.2x 20-period average to avoid low-volume noise.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_rsi_momentum_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI (14-period) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:15])
            avg_loss[i] = np.mean(loss[0:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1h = 100 - (100 / (1 + rs))
    
    # 4h RSI (14-period) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    
    avg_gain_4h = np.full(len(close_4h), np.nan)
    avg_loss_4h = np.full(len(close_4h), np.nan)
    for i in range(14, len(close_4h)):
        if i == 14:
            avg_gain_4h[i] = np.mean(gain_4h[0:15])
            avg_loss_4h[i] = np.mean(loss_4h[0:15])
        else:
            avg_gain_4h[i] = (avg_gain_4h[i-1] * 13 + gain_4h[i]) / 14
            avg_loss_4h[i] = (avg_loss_4h[i-1] * 13 + loss_4h[i]) / 14
    
    rs_4h = np.where(avg_loss_4h != 0, avg_gain_4h / avg_loss_4h, 0)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Volume filter: current volume > 1.2x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i+1])
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(rsi_1h[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session and volume filters
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        volume_filter = volume[i] > vol_ma[i] * 1.2
        
        if not (in_session and volume_filter):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI < 40 (momentum fade) or opposite 4h signal
            if (rsi_1h[i] < 40 or rsi_4h_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI > 60 (momentum fade) or opposite 4h signal
            if (rsi_1h[i] > 60 or rsi_4h_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extreme in direction of 4h trend
            # Long: 1h RSI < 30 (oversold) and 4h RSI > 50 (uptrend)
            if (rsi_1h[i] < 30 and rsi_4h_aligned[i] > 50):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: 1h RSI > 70 (overbought) and 4h RSI < 50 (downtrend)
            elif (rsi_1h[i] > 70 and rsi_4h_aligned[i] < 50):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals