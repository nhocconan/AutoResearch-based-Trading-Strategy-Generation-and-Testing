#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_PriceActionReversal_V1"
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
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d SMA(50) for trend filter
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d RSI(14) for momentum
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 50.0), where=avg_loss!=0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align indicators to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Price action patterns: inside bar and outside bar detection
    # Inside bar: current high < prev high AND current low > prev low
    inside_bar = (high[1:] < high[:-1]) & (low[1:] > low[:-1])
    inside_bar = np.concatenate([[False], inside_bar])
    
    # Outside bar: current high > prev high AND current low < prev low
    outside_bar = (high[1:] > high[:-1]) & (low[1:] < low[:-1])
    outside_bar = np.concatenate([[False], outside_bar])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        if np.isnan(atr_14_aligned[i]) or np.isnan(sma_50_aligned[i]) or \
           np.isnan(rsi_14_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_14_aligned[i]
        sma = sma_50_aligned[i]
        rsi = rsi_14_aligned[i]
        ib = inside_bar[i]
        ob = outside_bar[i]
        
        if position == 0:
            # Long setup: price above SMA, RSI not overbought, outside bar breakout
            if price > sma and rsi < 70 and ob:
                signals[i] = 0.25
                position = 1
            # Short setup: price below SMA, RSI not oversold, outside bar breakdown
            elif price < sma and rsi > 30 and ob:
                signals[i] = -0.25
                position = -1
            # Mean reversion from inside bar: fade the break of inside bar range
            elif ib:
                # Long if price breaks above inside bar high with momentum
                if price > high[i-1] and rsi > 50:
                    signals[i] = 0.25
                    position = 1
                # Short if price breaks below inside bar low with momentum
                elif price < low[i-1] and rsi < 50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions: RSI overbought or price closes below SMA
            if rsi > 70 or price < sma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: RSI oversold or price closes above SMA
            if rsi < 30 or price > sma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals