# 1d_KAMA_RSI_ChopFilter  
# Hypothesis: Daily trend following using Kaufman Adaptive Moving Average (KAMA) for direction,  
# RSI for momentum confirmation, and Choppiness Index (CHOP) to avoid ranging markets.  
# Long when KAMA slope > 0, RSI > 50, and CHOP > 61.8 (trending regime).  
# Short when KAMA slope < 0, RSI < 50, and CHOP > 61.8.  
# Uses discrete position sizing (0.30) to limit drawdown and reduce trade frequency.  
# Designed for low trade frequency (<25/year) to minimize fee impact in bear markets.  

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 as long-term trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Load daily data ONCE for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # placeholder, will fix below
    # Correct calculation of volatility (sum of absolute changes over 10 periods)
    volatility = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 10:
            volatility[i] = np.nan
        else:
            volatility[i] = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily (already daily, but for consistency)
    kama_aligned = kama  # already on daily timeframe
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP) on daily
    # True Range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max and min over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # CHOP formula
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    # Align indicators to daily (already aligned)
    rsi_aligned = rsi
    chop_aligned = chop
    ema200_1w_daily = ema200_1w_aligned  # weekly EMA200 aligned to daily
    
    # Prepare price and volume arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Signals array
    signals = np.zeros(n)
    
    # State tracking
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Main loop
    for i in range(100, n):  # Start after warmup period
        # Skip if any key indicator is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or \
           np.isnan(chop_aligned[i]) or np.isnan(ema200_1w_daily[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current values
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema200_val = ema200_1w_daily[i]
        
        # Determine KAMA slope (using 1-period change)
        if i > 0:
            kama_slope = kama_val - kama_aligned[i-1]
        else:
            kama_slope = 0
        
        # Entry conditions
        if position == 0:
            # Long: KAMA rising, RSI > 50, CHOP > 61.8 (trending), price above weekly EMA200
            if kama_slope > 0 and rsi_val > 50 and chop_val > 61.8 and price > ema200_val:
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: KAMA falling, RSI < 50, CHOP > 61.8 (trending), price below weekly EMA200
            elif kama_slope < 0 and rsi_val < 50 and chop_val > 61.8 and price < ema200_val:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        # Exit conditions
        elif position == 1:
            # Exit long: KAMA slope turns negative OR RSI < 45
            if kama_slope < 0 or rsi_val < 45:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: KAMA slope turns positive OR RSI > 55
            if kama_slope > 0 or rsi_val > 55:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0