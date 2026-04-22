#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1-day data once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 1-day ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike detection (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1D data to 4H timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any data is not ready
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi = rsi_aligned[i]
        atr = atr_aligned[i]
        ema50 = ema50_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) + price above EMA50 + volume spike
            if rsi < 30 and price > ema50 and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + price below EMA50 + volume spike
            elif rsi > 70 and price < ema50 and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: RSI returns to neutral zone (40-60) or ATR-based stop
            if position == 1:
                # Long exit: RSI > 40 or price drops below EMA50 - 1*ATR
                if rsi > 40 or price < ema50 - atr:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Short exit: RSI < 60 or price rises above EMA50 + 1*ATR
                if rsi < 60 or price > ema50 + atr:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_RSI_MeanReversion_EMA50_Volume"
timeframe = "4h"
leverage = 1.0