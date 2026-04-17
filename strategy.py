# 12h_KAMA_Direction_RSI_Filter_VolumeConfirm
# 12h timeframe with KAMA trend, RSI momentum filter, and volume confirmation
# Designed for low trade frequency and robustness in bull/bear markets
# Target: 50-150 total trades over 4 years (12-37/year)

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) for trend
    # Using ER (efficiency ratio) with 10 period
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume SMA(20) for volume confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 50)  # need sufficient data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        ema_50_val = ema_50_1d_aligned[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: Price above KAMA (uptrend) + RSI > 50 (bullish momentum) + 
            # Price above 1d EMA50 (long-term trend) + Volume confirmation
            if (price > kama_val and rsi_val > 50 and price > ema_50_val and 
                vol > 1.5 * vol_sma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend) + RSI < 50 (bearish momentum) + 
            # Price below 1d EMA50 (long-term trend) + Volume confirmation
            elif (price < kama_val and rsi_val < 50 and price < ema_50_val and 
                  vol > 1.5 * vol_sma_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price below KAMA (trend change) or RSI < 40 (momentum loss)
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price above KAMA (trend change) or RSI > 60 (momentum loss)
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Direction_RSI_Filter_VolumeConfirm"
timeframe = "12h"
leverage = 1.0