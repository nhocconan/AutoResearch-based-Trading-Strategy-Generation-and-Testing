#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    # Load daily data once for calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = (100 - (100 / (1 + rs))).values
    
    # Align daily indicators to 15m timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):
        # Skip if any data is not ready
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_aligned[i]
        rsi = rsi_aligned[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: RSI < 30 (oversold) + price > weekly EMA50 + volume spike
            if rsi < 30 and price > ema50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: RSI > 70 (overbought) + price < weekly EMA50 + volume spike
            elif rsi > 70 and price < ema50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: RSI returns to neutral range or volatility drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI > 50 or volatility drops significantly
                if rsi > 50 or atr < 0.5 * atr_aligned[i-1]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI < 50 or volatility drops significantly
                if rsi < 50 or atr < 0.5 * atr_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_RSI_Extreme_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0