# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour KAMA trend with daily RSI and volume confirmation
# Long when KAMA turns up, RSI < 50 (mean reversion in uptrend), and volume > average
# Short when KAMA turns down, RSI > 50 (mean reversion in downtrend), and volume > average
# Exit when KAMA reverses direction
# Uses KAMA's adaptive smoothing to reduce whipsaw in choppy markets
# Volume filter ensures trades occur with participation
# Target: 50-150 trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate KAMA (2-period ER, 30-period smoothing constant)
    # ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Pad volatility for first 10 values
    volatility[:10] = volatility[10] if len(volatility) > 10 else 0
    er = np.zeros_like(close)
    er[10:] = change[10:] / (volatility[10:] - volatility[:-10])
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate daily RSI (14-period)
    close_daily = df_daily['close'].values
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_daily = 100 - (100 / (1 + rs))
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    rsi_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi_daily)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for KAMA and RSI calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_daily_aligned[i]) or 
            np.isnan(vol_ma_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]  # Current 4h volume
        kama_prev = kama_aligned[i-1]
        kama_curr = kama_aligned[i]
        
        if position == 0:
            # Long setup: KAMA turning up, RSI < 50 (buy dip in uptrend), volume above average
            if (kama_curr > kama_prev and 
                rsi_daily_aligned[i] < 50 and 
                vol_4h_current > vol_ma_daily_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: KAMA turning down, RSI > 50 (sell rally in downtrend), volume above average
            elif (kama_curr < kama_prev and 
                  rsi_daily_aligned[i] > 50 and 
                  vol_4h_current > vol_ma_daily_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA turns down
            if kama_curr < kama_prev:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: KAMA turns up
            if kama_curr > kama_prev:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_KAMA_DailyRSI_Volume"
timeframe = "4h"
leverage = 1.0