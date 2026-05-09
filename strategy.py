#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI_Divergence_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.concatenate([np.full(14, np.nan), rsi_1d])
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6-period RSI for divergence detection
    delta_6 = np.diff(close)
    gain_6 = np.where(delta_6 > 0, delta_6, 0)
    loss_6 = np.where(delta_6 < 0, -delta_6, 0)
    avg_gain_6 = pd.Series(gain_6).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss_6 = pd.Series(loss_6).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs_6 = avg_gain_6 / (avg_loss_6 + 1e-10)
    rsi_6 = 100 - (100 / (1 + rs_6))
    rsi_6 = np.concatenate([np.full(6, np.nan), rsi_6])
    
    # Align 1d indicators to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter: current volume > 1.8 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(rsi_6[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_1d_val = rsi_1d_aligned[i]
        ema50 = ema50_1d_aligned[i]
        rsi_6_val = rsi_6[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            # Plus 1d uptrend (price > EMA50) and volume spike
            if (i >= 2 and 
                close[i] < close[i-1] and 
                close[i-1] < close[i-2] and
                rsi_6_val > rsi_6[i-1] and 
                rsi_6[i-1] > rsi_6[i-2] and
                close[i] > ema50 and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Bearish divergence: price makes higher high, RSI makes lower high
            # Plus 1d downtrend (price < EMA50) and volume spike
            elif (i >= 2 and 
                  close[i] > close[i-1] and 
                  close[i-1] > close[i-2] and
                  rsi_6_val < rsi_6[i-1] and 
                  rsi_6[i-1] < rsi_6[i-2] and
                  close[i] < ema50 and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend turns down
            if rsi_6_val > 70 or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold or trend turns up
            if rsi_6_val < 30 or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals