#!/usr/bin/env python3
"""
4h_RSI_Extreme_Top_Bottom_Reversal_1dTrend_Volume
Hypothesis: RSI extremes (above 80 or below 20) combined with 1d EMA50 trend filter and volume spikes capture exhaustion moves in both bull and bear markets. 
The strategy avoids choppy markets by requiring trend alignment and uses volume to confirm institutional participation. 
Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag while capturing high-probability reversals.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI, EMA, and volume MA
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        rsi_val = rsi[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: RSI below 20 (oversold) with uptrend and volume spike
            if rsi_val < 20 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: RSI above 80 (overbought) with downtrend and volume spike
            elif rsi_val > 80 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 50 or trend turns down
            if rsi_val > 50 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI crosses below 50 or trend turns up
            if rsi_val < 50 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RSI_Extreme_Top_Bottom_Reversal_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0