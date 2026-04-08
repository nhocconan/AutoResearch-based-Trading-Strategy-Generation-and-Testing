#!/usr/bin/env python3
# [24994] 1h_4h_1d_rsi_momentum_v1
# Hypothesis: On 1h, use 4h RSI for trend bias (bullish >50, bearish <50) and 1d RSI for regime filter (avoid extremes).
# Enter long when 1h RSI crosses above 30 (from oversold) AND 4h RSI > 50 AND 1d RSI between 30 and 70.
# Enter short when 1h RSI crosses below 70 (from overbought) AND 4h RSI < 50 AND 1d RSI between 30 and 70.
# Exit when 1h RSI returns to 50 (mean reversion) or opposite signal.
# Uses 4h/1d for direction/regime, 1h only for entry timing. Target: 15-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_rsi_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1h = 100 - (100 / (1 + rs))
    
    # Get 4h data for RSI trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    # Calculate 4h RSI (14-period)
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = np.zeros(len(close_4h))
    avg_loss_4h = np.zeros(len(close_4h))
    for i in range(1, len(close_4h)):
        avg_gain_4h[i] = (avg_gain_4h[i-1] * 13 + gain_4h[i]) / 14
        avg_loss_4h[i] = (avg_loss_4h[i-1] * 13 + loss_4h[i]) / 14
    rs_4h = np.divide(avg_gain_4h, avg_loss_4h, out=np.zeros_like(avg_gain_4h), where=avg_loss_4h!=0)
    rsi_4h_raw = 100 - (100 / (1 + rs_4h))
    # Align 4h RSI to 1h timeframe (wait for 4h bar close)
    rsi_4h = align_htf_to_ltf(prices, df_4h, rsi_4h_raw)
    
    # Get 1d data for RSI regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # Calculate 1d RSI (14-period)
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = np.zeros(len(close_1d))
    avg_loss_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain_1d[i]) / 14
        avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss_1d[i]) / 14
    rs_1d = np.divide(avg_gain_1d, avg_loss_1d, out=np.zeros_like(avg_gain_1d), where=avg_loss_1d!=0)
    rsi_1d_raw = 100 - (100 / (1 + rs_1d))
    # Align 1d RSI to 1h timeframe (wait for 1d bar close)
    rsi_1d = align_htf_to_ltf(prices, df_1d, rsi_1d_raw)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if data not ready
        if (np.isnan(rsi_1h[i]) or np.isnan(rsi_4h[i]) or np.isnan(rsi_1d[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        rsi1 = rsi_1h[i]
        rsi4 = rsi_4h[i]
        rsid = rsi_1d[i]
        
        if position == 1:  # Long
            # Exit: RSI returns to 50 (mean reversion) or bearish flip
            if rsi1 <= 50 or rsi4 < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: RSI returns to 50 (mean reversion) or bullish flip
            if rsi1 >= 50 or rsi4 > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: 1h RSI crosses above 30 (from oversold) AND 4h RSI > 50 AND 1d RSI in neutral zone
            if rsi1 > 30 and rsi_1h[i-1] <= 30 and rsi4 > 50 and 30 <= rsid <= 70:
                position = 1
                signals[i] = 0.20
            # Enter short: 1h RSI crosses below 70 (from overbought) AND 4h RSI < 50 AND 1d RSI in neutral zone
            elif rsi1 < 70 and rsi_1h[i-1] >= 70 and rsi4 < 50 and 30 <= rsid <= 70:
                position = -1
                signals[i] = -0.20
    
    return signals