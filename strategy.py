#!/usr/bin/env python3

"""
Hypothesis: 1-hour RSI(14) mean reversion with 4-hour RSI(14) trend filter and volume spike confirmation.
Goes long when 1h RSI < 30 (oversold) and 4h RSI > 50 (uptrend), short when 1h RSI > 70 (overbought) and 4h RSI < 50 (downtrend).
Requires volume > 1.3x 20-period average for entry confirmation.
Exits when RSI returns to neutral (40-60 range) or opposite extreme.
Targets 15-37 trades/year (60-150 total over 4 years) with disciplined entries to avoid fee drag.
Uses 4h trend to avoid counter-trend trades in both bull and bear markets.
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
    
    # Load 4h data for RSI trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h RSI(14) for trend filter
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_4h = 100 - (100 / (1 + rs))
    rsi_14_4h = rsi_14_4h.values
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # Calculate 1h RSI(14) for entry signals
    delta_1h = np.diff(close, prepend=close[0])
    gain_1h = np.where(delta_1h > 0, delta_1h, 0)
    loss_1h = np.where(delta_1h < 0, -delta_1h, 0)
    avg_gain_1h = pd.Series(gain_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1h = pd.Series(loss_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1h = avg_gain_1h / (avg_loss_1h + 1e-10)
    rsi_14_1h = 100 - (100 / (1 + rs_1h))
    rsi_14_1h = rsi_14_1h.values
    
    # Volume spike: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(rsi_14_4h_aligned[i]) or np.isnan(rsi_14_1h[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: 1h RSI oversold (<30) and 4h RSI uptrend (>50)
            if rsi_14_1h[i] < 30 and rsi_14_4h_aligned[i] > 50:
                signals[i] = 0.20
                position = 1
            # Short: 1h RSI overbought (>70) and 4h RSI downtrend (<50)
            elif rsi_14_1h[i] > 70 and rsi_14_4h_aligned[i] < 50:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral range (40-60) or hits opposite extreme
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI returns to neutral or becomes overbought
                if rsi_14_1h[i] >= 40 or rsi_14_1h[i] > 70:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI returns to neutral or becomes oversold
                if rsi_14_1h[i] <= 60 or rsi_14_1h[i] < 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI14_MeanReversion_4hRSITrend_Volume"
timeframe = "1h"
leverage = 1.0