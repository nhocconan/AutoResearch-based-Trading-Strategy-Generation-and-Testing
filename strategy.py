#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI pullback and volume confirmation
# - KAMA (10,2,30) identifies trend direction
# - RSI(14) pulls back to 40-60 during trend for entry
# - Volume > 1.5x 20-period average confirms momentum
# - Weekly trend filter avoids counter-trend trades
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation (ER=10, FAST=2, SLOW=30)
    change = np.abs(np.diff(close, k=10))
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Initialize
    for i in range(10, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI calculation
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price > KAMA, RSI 40-60, volume surge, weekly uptrend
            if (price > kama[i] and 
                40 <= rsi[i] <= 60 and 
                vol > 1.5 * vol_ma[i] and 
                ema_34_1w_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI 40-60, volume surge, weekly downtrend
            elif (price < kama[i] and 
                  40 <= rsi[i] <= 60 and 
                  vol > 1.5 * vol_ma[i] and 
                  ema_34_1w_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < KAMA or weekly trend turns down
            if price < kama[i] or ema_34_1w_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA or weekly trend turns up
            if price > kama[i] or ema_34_1w_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0