# Your turn. Make it count.
#!/usr/bin/env python3
"""
4h_RSI_Divergence_Volume_Pattern
Hypothesis: RSI divergence with volume pattern on 4h chart identifies trend exhaustion and reversals.
Works in both bull and bear markets by capturing exhaustion moves. Uses RSI(14) for divergence detection,
volume confirmation for conviction, and avoids overtrading through strict divergence criteria.
Target: 20-40 trades/year per symbol with strong risk-adjusted returns.
"""

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
    
    # RSI calculation with proper smoothing
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Use Wilder's smoothing (alpha = 1/period)
        avg_gain = np.full_like(prices, np.nan)
        avg_loss = np.full_like(prices, np.nan)
        
        if len(prices) > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            alpha = 1.0 / period
            for i in range(period + 1, len(prices)):
                avg_gain[i] = alpha * gain[i-1] + (1 - alpha) * avg_gain[i-1]
                avg_loss[i] = alpha * loss[i-1] + (1 - alpha) * avg_loss[i-1]
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate RSI
    rsi = calculate_rsi(close, 14)
    
    # Volume spike detection: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Price action patterns for divergence confirmation
    # Higher high in price with lower high in RSI = bearish divergence
    # Lower low in price with higher low in RSI = bullish divergence
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for RSI and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
            
        # Look back for divergence patterns (check last 10 bars for efficiency)
        lookback = min(10, i)
        bullish_div = False
        bearish_div = False
        
        if i >= lookback:
            # Check for bullish divergence: lower low in price, higher low in RSI
            price_lows = []
            rsi_lows = []
            for j in range(i - lookback + 1, i + 1):
                if j > 0 and low[j] <= low[j-1] and low[j] <= low[j+1 if j+1 < i else j]:
                    price_lows.append((j, low[j]))
                    rsi_lows.append((j, rsi[j]))
            
            if len(price_lows) >= 2:
                # Check last two lows for divergence
                if (price_lows[-1][1] < price_lows[-2][1] and 
                    rsi_lows[-1][1] > rsi_lows[-2][1]):
                    bullish_div = True
            
            # Check for bearish divergence: higher high in price, lower high in RSI
            price_highs = []
            rsi_highs = []
            for j in range(i - lookback + 1, i + 1):
                if j > 0 and high[j] >= high[j-1] and high[j] >= high[j+1 if j+1 < i else j]:
                    price_highs.append((j, high[j]))
                    rsi_highs.append((j, rsi[j]))
            
            if len(price_highs) >= 2:
                # Check last two highs for divergence
                if (price_highs[-1][1] > price_highs[-2][1] and 
                    rsi_highs[-1][1] < rsi_highs[-2][1]):
                    bearish_div = True
        
        if position == 0:
            # Enter long on bullish divergence with volume spike
            if bullish_div and vol_spike[i] and rsi[i] < 40:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish divergence with volume spike
            elif bearish_div and vol_spike[i] and rsi[i] > 60:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on bearish divergence or RSI overbought
            if bearish_div or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on bullish divergence or RSI oversold
            if bullish_div or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Divergence_Volume_Pattern"
timeframe = "4h"
leverage = 1.0