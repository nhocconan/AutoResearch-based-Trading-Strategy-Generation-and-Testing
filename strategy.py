#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA with RSI and chop filter
# KAMA adapts to market noise - fast in trends, slow in ranging markets
# Use 1d KAMA direction (trend) + RSI(14) for entry timing + chop filter to avoid whipsaws
# Targets 30-100 total trades over 4 years (7-25/year) by using strict daily timeframe
# Works in bull (follow KAMA trend) and bear (RSI extremes with chop filter)

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    def kama(price, er_period=10, fast=2, slow=30):
        # Change and volatility
        change = np.abs(np.diff(price, prepend=price[0]))
        volatility = np.abs(np.diff(price, prepend=price[0]))
        
        # Efficiency Ratio
        er = np.zeros_like(price)
        for i in range(er_period, len(price)):
            price_change = np.abs(price[i] - price[i-er_period])
            price_volatility = np.sum(volatility[i-er_period+1:i+1])
            if price_volatility > 0:
                er[i] = price_change / price_volatility
            else:
                er[i] = 1.0
        
        # Smoothing Constants
        sc = np.zeros_like(price)
        fast_sc = 2.0 / (fast + 1)
        slow_sc = 2.0 / (slow + 1)
        sc = er * (fast_sc - slow_sc) + slow_sc
        sc = sc * sc  # Square for smoothing
        
        # KAMA calculation
        kama_val = np.zeros_like(price)
        kama_val[0] = price[0]
        for i in range(1, len(price)):
            kama_val[i] = kama_val[i-1] + sc[i] * (price[i] - kama_val[i-1])
        
        return kama_val
    
    kama_val = kama(close, 10, 2, 30)
    
    # RSI(14)
    def rsi(price, period=14):
        delta = np.diff(price, prepend=price[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        
        # Initial average
        if len(price) >= period:
            avg_gain[period-1] = np.mean(gain[1:period])
            avg_loss[period-1] = np.mean(loss[1:period])
            
            # Wilder smoothing
            for i in range(period, len(price)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_val = rsi(close, 14)
    
    # Choppiness Index (CHOP) - measures ranging vs trending
    def chop(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR
        atr = np.zeros_like(close)
        for i in range(period, len(close)):
            if i == period:
                atr[i] = np.nanmean(tr[1:i+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Highest high and lowest low over period
        hh = np.zeros_like(high)
        ll = np.zeros_like(low)
        for i in range(period-1, len(high)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop_val = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if np.sum(atr[i-period+1:i+1]) > 0 and hh[i] > ll[i]:
                chop_val[i] = 100 * np.log10(np.sum(atr[i-period+1:i+1]) / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop_val[i] = 50.0
        
        return chop_val
    
    chop_val = chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or np.isnan(chop_val[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # KAMA direction: slope of KAMA
        if i >= 2:
            kama_slope = kama_val[i] - kama_val[i-2]
        else:
            kama_slope = 0
        
        # Chop regime: > 50 = ranging, < 50 = trending
        is_ranging = chop_val[i] > 50
        
        if position == 1:  # long position
            # Exit: RSI > 70 (overbought) or KAMA turns down
            if rsi_val[i] > 70 or kama_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI < 30 (oversold) or KAMA turns up
            if rsi_val[i] < 30 or kama_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on regime
            if is_ranging:
                # Ranging market: mean reversion at RSI extremes
                if rsi_val[i] < 30 and kama_slope > 0:  # Oversold and KAMA turning up
                    signals[i] = 0.25
                    position = 1
                elif rsi_val[i] > 70 and kama_slope < 0:  # Overbought and KAMA turning down
                    signals[i] = -0.25
                    position = -1
            else:
                # Trending market: follow KAMA direction
                if kama_slope > 0 and rsi_val[i] > 50:  # Uptrend and not overbought
                    signals[i] = 0.25
                    position = 1
                elif kama_slope < 0 and rsi_val[i] < 50:  # Downtrend and not oversold
                    signals[i] = -0.25
                    position = -1
    
    return signals