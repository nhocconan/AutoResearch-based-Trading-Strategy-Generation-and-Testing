#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA with RSI filter and 1d trend confirmation.
# KAMA adapts to market noise, reducing whipsaws in sideways markets.
# Long when KAMA rising, RSI > 50, and 1d EMA50 uptrend.
# Short when KAMA falling, RSI < 50, and 1d EMA50 downtrend.
# Uses discrete position sizing (0.25) to minimize churn.
# Designed to work in both bull (trend following) and bear (mean reversion via RSI extremes) markets.
name = "12h_KAMA_RSI_1dEMA50"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # KAMA calculation (ER=10, SC=30)
    def calculate_kama(close, er_len=10, sc_len=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        vol = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
        # Correct volatility calculation: sum of absolute changes over er_len period
        vol = np.zeros_like(close)
        for i in range(len(close)):
            if i < er_len:
                vol[i] = np.nan
            else:
                vol[i] = np.sum(np.abs(np.diff(close[i-er_len:i+1])))
        er = np.where(vol != 0, change / vol, 0)
        sc = np.power(er * (2/(sc_len+1) - 2/(er_len+1)) + 2/(er_len+1), 2)
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if np.isnan(sc[i]):
                kama[i] = kama[i-1]
            else:
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 30)
    
    # RSI calculation (14-period)
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Align 1d EMA50 to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # KAMA slope (1-period change)
    kama_slope = np.diff(kama, prepend=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure EMA50, KAMA, RSI are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        ema_50_val = ema_50_aligned[i]
        kama_slope_val = kama_slope[i]
        
        if position == 0:
            # Enter long if KAMA rising, RSI > 50, and price above 1d EMA50
            if kama_slope_val > 0 and rsi_val > 50 and price > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Enter short if KAMA falling, RSI < 50, and price below 1d EMA50
            elif kama_slope_val < 0 and rsi_val < 50 and price < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when KAMA falls or RSI < 50
            if kama_slope_val <= 0 or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when KAMA rises or RSI > 50
            if kama_slope_val >= 0 or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals