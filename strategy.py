#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Direction_RSI_Pullback"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on daily close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, 10))
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(kama_4h[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama_val = kama_4h[i]
        rsi_val = rsi[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        kama_bullish = price > kama_val
        kama_bearish = price < kama_val
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        
        if position == 0:
            # Long: Price above KAMA (bullish trend) + RSI oversold + volume
            if kama_bullish and rsi_oversold and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (bearish trend) + RSI overbought + volume
            elif kama_bearish and rsi_overbought and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below KAMA OR RSI overbought
            if not kama_bullish or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above KAMA OR RSI oversold
            if not kama_bearish or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals