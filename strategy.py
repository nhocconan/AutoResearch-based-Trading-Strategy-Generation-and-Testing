#!/usr/bin/env python3
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
    
    # Get daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14d = np.zeros(len(df_1d))
    for i in range(len(tr)):
        if i < 13:
            atr_14d[i] = np.mean(tr[:i+1]) if i > 0 else tr[i]
        else:
            atr_14d[i] = (atr_14d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to 4h timeframe
    atr_14d_aligned = align_htf_to_ltf(prices, df_1d, atr_14d)
    
    # Calculate 4h ATR(14) for stop loss and position sizing
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[high[0] - low[0]], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    
    atr_14 = np.zeros(n)
    for i in range(n):
        if i < 13:
            atr_14[i] = np.mean(tr_4h[:i+1]) if i > 0 else tr_4h[i]
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr_4h[i]) / 14
    
    # Calculate 4h EMA(50) for trend filter
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < 1:
            avg_gain[i] = gain[i]
            avg_loss[i] = loss[i]
        elif i < 14:
            avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
            avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(atr_14d_aligned[i]) or np.isnan(ema_50[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_4h = atr_14[i]
        daily_volatility = atr_14d_aligned[i]
        
        # Skip if daily volatility is too low (avoid choppy markets)
        if daily_volatility <= 0:
            signals[i] = 0.0
            continue
        
        # Normalize price change by daily volatility for regime detection
        price_change = abs(close[i] - close[i-1]) if i > 0 else 0
        volatility_ratio = price_change / daily_volatility
        
        # Volatility filter: only trade when volatility is elevated (> 0.5x daily ATR)
        volatility_filter = volatility_ratio > 0.5
        
        if position == 0:
            # Long signal: RSI < 30 (oversold) + price above EMA50 (uptrend bias) + volatility filter
            if (rsi[i] < 30 and 
                price > ema_50[i] and 
                volatility_filter):
                # Size inversely proportional to volatility (smaller size in high vol)
                vol_adjustment = min(1.0, daily_volatility / (atr_4h * 2))  # Normalize by 4h ATR
                size = 0.25 * vol_adjustment
                signals[i] = size
                position = 1
            # Short signal: RSI > 70 (overbought) + price below EMA50 (downtrend bias) + volatility filter
            elif (rsi[i] > 70 and 
                  price < ema_50[i] and 
                  volatility_filter):
                vol_adjustment = min(1.0, daily_volatility / (atr_4h * 2))
                size = 0.25 * vol_adjustment
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 70 (overbought) or price below EMA50
            if rsi[i] > 70 or price < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: RSI < 30 (oversold) or price above EMA50
            if rsi[i] < 30 or price > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_RSI_EMA50_VolatilityFilter_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0