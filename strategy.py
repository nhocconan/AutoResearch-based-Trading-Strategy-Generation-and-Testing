#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily 20-period EMA for trend
    close_1d = df_1d['close'].values
    ema_20_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        multiplier = 2 / (20 + 1)
        ema_20_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_20_1d[i] = (close_1d[i] * multiplier) + (ema_20_1d[i-1] * (1 - multiplier))
    
    # Calculate daily 14-period RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14_1d = 100 - (100 / (1 + rs))
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_1d[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily indicators to daily timeframe (1:1 mapping)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price above EMA20, RSI oversold (<30), and low volatility (ATR below median)
            if (price > ema_20_1d_aligned[i] and 
                rsi_14_1d_aligned[i] < 30 and
                atr_14_1d_aligned[i] < np.nanmedian(atr_14_1d_aligned[max(0, i-30):i])):
                signals[i] = size
                position = 1
            # Short: Price below EMA20, RSI overbought (>70), and low volatility
            elif (price < ema_20_1d_aligned[i] and 
                  rsi_14_1d_aligned[i] > 70 and
                  atr_14_1d_aligned[i] < np.nanmedian(atr_14_1d_aligned[max(0, i-30):i])):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below EMA20 or RSI overbought
            if price < ema_20_1d_aligned[i] or rsi_14_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above EMA20 or RSI oversold
            if price > ema_20_1d_aligned[i] or rsi_14_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_EMA20_RSI14_VolatilityFilter"
timeframe = "1d"
leverage = 1.0