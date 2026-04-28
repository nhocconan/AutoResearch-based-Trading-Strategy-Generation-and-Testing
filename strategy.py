# 1d_KAMA_RSI_ChopFilter
# Hypothesis: KAMA adapts to market conditions, RSI identifies overbought/oversold, and Choppiness Index filters ranging markets. Works in bull/bear by adapting to volatility and avoiding false signals in chop.
# Target: 10-25 trades/year on 1d timeframe to minimize fee drag.
# Uses KAMA direction, RSI extremes, and Choppiness Index regime filter.

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20) for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly trend to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i-1] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Calculate Choppiness Index(14)
    atr = np.zeros_like(close)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr[1:] = np.sum(tr.reshape(-1, 14), axis=1) / 14
    atr = np.concatenate([np.full(14, np.nan), atr])
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(tr.reshape(-1, 14), axis=1) / 14 / (max_high - min_low)) / np.log10(14)
    chop = np.concatenate([np.full(14, np.nan), chop])
    
    # Align weekly trend to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop[i]) or
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: avoid choppy markets (Choppiness > 61.8)
        if chop[i] > 61.8:
            signals[i] = 0.0
            continue
        
        # KAMA direction: price above/below KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # RSI extremes: oversold/overbought
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Weekly trend filter: align with higher timeframe
        uptrend = close[i] > ema20_1w_aligned[i]
        downtrend = close[i] < ema20_1w_aligned[i]
        
        # Entry conditions
        long_entry = above_kama and rsi_oversold and uptrend
        short_entry = below_kama and rsi_overbought and downtrend
        
        # Exit conditions: opposite signal or RSI normalization
        long_exit = below_kama or rsi[i] > 50
        short_exit = above_kama or rsi[i] < 50
        
        if long_entry:
            signals[i] = 0.25
        elif short_entry:
            signals[i] = -0.25
        elif long_exit and i > 0 and signals[i-1] > 0:
            signals[i] = 0.0
        elif short_exit and i > 0 and signals[i-1] < 0:
            signals[i] = 0.0
        else:
            # Hold previous position
            signals[i] = signals[i-1]
    
    return signals

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0