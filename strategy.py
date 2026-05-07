#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour MACD histogram divergence with 4-hour RSI filter and volume confirmation.
# Long when: MACD histogram crosses above zero AND 4h RSI < 40 AND volume > 1.5 * EMA20(volume).
# Short when: MACD histogram crosses below zero AND 4h RSI > 60 AND volume > 1.5 * EMA20(volume).
# Uses MACD for momentum reversal, 4h RSI for overbought/oversold conditions, volume for confirmation.
# Designed for low trade frequency (target: 15-37/year) to minimize fee drag.
# Works in bull markets via bullish reversals and in bear markets via bearish reversals.
name = "1h_MACD_RSI_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # MACD: EMA12 - EMA26, Signal = EMA9 of MACD
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd = ema_12 - ema_26
    signal_line = pd.Series(macd).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd - signal_line
    
    # Load 4h data for RSI
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # RSI calculation (14-period)
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(macd_hist[i]) or np.isnan(rsi_4h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: MACD hist crosses above zero AND RSI < 40 AND volume spike
            long_condition = (macd_hist[i] > 0 and macd_hist[i-1] <= 0) and (rsi_4h_aligned[i] < 40) and volume_spike[i]
            # Short: MACD hist crosses below zero AND RSI > 60 AND volume spike
            short_condition = (macd_hist[i] < 0 and macd_hist[i-1] >= 0) and (rsi_4h_aligned[i] > 60) and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: MACD hist crosses below zero or RSI > 70
            if (macd_hist[i] < 0 and macd_hist[i-1] >= 0) or rsi_4h_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: MACD hist crosses above zero or RSI < 30
            if (macd_hist[i] > 0 and macd_hist[i-1] <= 0) or rsi_4h_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals