#!/usr/bin/env python3
"""
1h_RSI14_Adaptive_Trend_Filter_v1
Hypothesis: RSI(14) with adaptive trend filter (4h EMA50) and volume confirmation captures momentum shifts while avoiding whipsaws. In bull markets, buys when RSI>50 and EMA50 uptrend; in bear markets, shorts when RSI<50 and EMA50 downtrend. Uses 1h timeframe for entry timing, 4h for trend filter to reduce trade frequency. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for RSI and volume MA
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema_trend = ema50_4h_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: RSI > 50 (bullish momentum) with uptrend and volume spike
            if rsi_val > 50 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: RSI < 50 (bearish momentum) with downtrend and volume spike
            elif rsi_val < 50 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI < 40 or trend turns down
            if rsi_val < 40 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI > 60 or trend turns up
            if rsi_val > 60 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI14_Adaptive_Trend_Filter_v1"
timeframe = "1h"
leverage = 1.0