#!/usr/bin/env python3
"""
4h RSI 2-Period Pullback with 1w Trend Filter and Volume Confirmation
Hypothesis: In strong trends (1w EMA50), 2-period RSI pulls back to oversold/overbought levels 
offer high-probability continuation entries. Works in bull markets via long pullbacks and 
in bear markets via short pullbacks. Volume confirmation filters weak moves. Low trade 
frequency due to strict RSI threshold and trend alignment requirement.
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
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # RSI(2) on close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        trend = ema50_1w_aligned[i]
        rsi_val = rsi_values[i]
        vol_ok = vol_confirm[i]
        
        if position == 0:
            # Enter long: RSI(2) < 10 (oversold) in uptrend with volume
            if rsi_val < 10 and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI(2) > 90 (overbought) in downtrend with volume
            elif rsi_val > 90 and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI(2) > 50 (neutral) or trend turns down
            if rsi_val > 50 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI(2) < 50 (neutral) or trend turns up
            if rsi_val < 50 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI2_Pullback_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0