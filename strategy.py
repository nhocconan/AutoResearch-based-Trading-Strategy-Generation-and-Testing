#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Close for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA (10) on 1d close - trend direction
    change_1d = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_1d = np.abs(np.diff(close_1d))
    er_1d = np.where(volatility_1d != 0, change_1d / volatility_1d, 0)
    sc_1d = (er_1d * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (close_1d[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # RSI(14) on 1d close - momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Choppiness Index(14) on 1d - regime filter
    atr_1d = np.zeros(len(close_1d))
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.max(high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0]))], tr_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop_1d = 100 * np.log10(
        (atr_1d * 14) / (max_high_1d - min_low_1d)
    ) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 12h price action
    sma_12h = pd.Series(close).rolling(window=12, min_periods=12).mean()
    
    signals = np.zeros(n)
    
    for i in range(12, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(sma_12h[i])):
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama_1d_aligned[i]
        price_below_kama = close[i] < kama_1d_aligned[i]
        
        # RSI conditions: not extreme
        rsi_not_overbought = rsi_1d_aligned[i] < 70
        rsi_not_oversold = rsi_1d_aligned[i] > 30
        
        # Chop regime: trending market (chop < 61.8)
        trending_market = chop_1d_aligned[i] < 61.8
        
        # Long: price above KAMA + not overbought + trending
        if price_above_kama and rsi_not_overbought and trending_market:
            signals[i] = 0.25
        
        # Short: price below KAMA + not oversold + trending
        elif price_below_kama and rsi_not_oversold and trending_market:
            signals[i] = -0.25
        
        # Exit: chop becomes too high (choppy/ranging) or price crosses KAMA in opposite direction
        elif chop_1d_aligned[i] >= 61.8:
            signals[i] = 0.0
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and price_below_kama) or
               (signals[i-1] == -0.25 and price_above_kama))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_KAMA_RSI_Chop"
timeframe = "12h"
leverage = 1.0