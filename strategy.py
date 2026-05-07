#!/usr/bin/env python3
name = "1d_KAMA_Trend_Filter_With_RSI_And_Volume"
timeframe = "1d"
leverage = 1.0

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
    volume = volumes = prices['volume'].values
    
    # Calculate KAMA on close prices
    def kama(price, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing Constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.full_like(price, np.nan, dtype=float)
        kama[period] = price[period]
        for i in range(period+1, len(price)):
            if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    # Calculate RSI
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate ATR
    def atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_50)
    
    # KAMA trend (10,2,30)
    kama_val = kama(close, 10, 2, 30)
    # RSI (14)
    rsi_val = rsi(close, 14)
    # ATR for stop (14)
    atr_val = atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure weekly EMA and KAMA/RSI warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or np.isnan(atr_val[i]) or 
            np.isnan(weekly_ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5 x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > 1.5 * vol_ma
        else:
            volume_filter = False
        
        if position == 0:
            # Long: Price above KAMA, RSI > 50, weekly uptrend, volume support
            if (close[i] > kama_val[i] and 
                rsi_val[i] > 50 and 
                close[i] > weekly_ema_50_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, RSI < 50, weekly downtrend, volume support
            elif (close[i] < kama_val[i] and 
                  rsi_val[i] < 50 and 
                  close[i] < weekly_ema_50_aligned[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions: reversal of entry signals
            if position == 1:
                # Exit long: price below KAMA OR RSI < 40
                if close[i] < kama_val[i] or rsi_val[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price above KAMA OR RSI > 60
                if close[i] > kama_val[i] or rsi_val[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals