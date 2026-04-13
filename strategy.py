#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA trend + RSI mean reversion + chop regime filter
    # Long: KAMA upward AND RSI < 40 (oversold in uptrend) AND chop < 61.8 (trending)
    # Short: KAMA downward AND RSI > 60 (overbought in downtrend) AND chop < 61.8 (trending)
    # Exit: RSI crosses 50 or chop > 61.8 (range) or opposite KAMA signal
    # Using 1d timeframe for low trade frequency (target 7-25/year), KAMA for adaptive trend,
    # RSI for mean reversion entries within trend, and chop filter to avoid whipsaws.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily KAMA(10,2,30) - adaptive trend
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        n = len(close)
        if n < er_length:
            return np.full(n, np.nan)
        
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.diff(close), 'shape') else np.nan
        # Proper volatility calculation: sum of absolute changes over er_length period
        volatility = np.full(n, np.nan)
        for i in range(er_length, n):
            volatility[i] = np.sum(np.abs(np.diff(close[i-er_length:i])))
        
        er = np.where(volatility > 0, change / volatility, 0)
        er = np.concatenate([np.full(er_length, np.nan), er])
        
        # Smoothing Constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama = np.full(n, np.nan)
        kama[er_length] = close[er_length]  # seed
        for i in range(er_length+1, n):
            if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    kama_up = kama > np.roll(kama, 1)  # upward slope
    kama_down = kama < np.roll(kama, 1)  # downward slope
    
    # Calculate daily RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        if len(close) > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    rsi_oversold = rsi < 40
    rsi_overbought = rsi > 60
    rsi_exit = (rsi > 50) & (rsi < 60)  # exit long when RSI > 50
    rsi_exit_short = (rsi < 50) & (rsi > 40)  # exit short when RSI < 50
    
    # Calculate daily Chopiness Index(14) for regime filter
    def calculate_chop(high, low, close, period=14):
        n = len(close)
        if n < period:
            return np.full(n, np.nan)
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # first period
        
        # Sum of TR over period
        tr_sum = np.full(n, np.nan)
        for i in range(period, n):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.full(n, np.nan)
        min_low = np.full(n, np.nan)
        for i in range(period-1, n):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop formula: 100 * log10(sum(tr) / (max_high - min_low)) / log10(period)
        denominator = max_high - min_low
        chop = np.full(n, np.nan)
        for i in range(period-1, n):
            if denominator[i] > 0:
                chop[i] = 100 * np.log10(tr_sum[i] / denominator[i]) / np.log10(period)
            else:
                chop[i] = 50  # neutral when no range
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_filter = chop < 61.8  # trending market (chop < 61.8)
    chop_exit = chop > 61.8   # exit when range/choppy
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Entry logic
        long_entry = kama_up[i] and rsi_oversold[i] and chop_filter[i]
        short_entry = kama_down[i] and rsi_overbought[i] and chop_filter[i]
        
        # Exit logic
        long_exit = rsi_exit[i] or chop_exit[i] or kama_down[i]
        short_exit = rsi_exit_short[i] or chop_exit[i] or kama_up[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_kama_rsi_chop_regime_v1"
timeframe = "1d"
leverage = 1.0