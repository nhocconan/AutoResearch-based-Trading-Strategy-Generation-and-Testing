#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_rsi_chop_filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for regime filter (choppiness)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA calculation for trend direction
    def calculate_kama(price, period=10, fast=2, slow=30):
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=0)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / volatility[period-1:]
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(price, np.nan)
        kama[period] = price[period]
        for i in range(period+1, len(price)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
        return kama
    
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # RSI calculation
    def calculate_rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Choppiness Index for regime filter
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        for i in range(period+1, len(close)):
            if i == period+1:
                atr[i] = np.mean(tr[1:i+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(period, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        
        chop = np.full_like(close, 50.0)
        for i in range(period, len(close)):
            if highest_high[i] != lowest_low[i]:
                chop[i] = 100 * np.log10(np.sum(atr[i-period+1:i+1]) / 
                                          np.log10(highest_high[i] - lowest_low[i]) * 
                                          (period / np.log10(period)))
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, period=14)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume filter
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma[:10] = np.nan
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_12h[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Choppiness regime: only trade when chop > 50 (ranging market)
        if chop_12h[i] <= 50:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 or KAMA turns down
            if rsi[i] > 70 or close[i] < kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 or KAMA turns up
            if rsi[i] < 30 or close[i] > kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: KAMA up + RSI < 30 (oversold) + volume
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: KAMA down + RSI > 70 (overbought) + volume
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals