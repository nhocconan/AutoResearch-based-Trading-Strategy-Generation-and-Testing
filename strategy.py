#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop regime filter
# KAMA adapts to market noise, reducing false signals in sideways markets.
# RSI provides mean-reversion signals when extremes occur.
# Chop filter identifies ranging markets (Chop > 61.8) for mean reversion.
# This combination should work in both bull (trend following via KAMA) and bear (mean reversion via RSI in chop) markets.
# Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (adaptive moving average) - trend component
    # Efficiency Ratio: |price change| / sum of absolute changes
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros(n)
    for i in range(1, n):
        if i >= 10:
            num = np.abs(close[i] - close[i-10])
            den = np.sum(abs_change[i-9:i+1])
            er[i] = num / den if den != 0 else 0
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA 2
    slow_sc = 2 / (30 + 1)  # for EMA 30
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14) - mean reversion component
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Chop index (14) - regime filter
    # Chop = 100 * log15(sum(TR) / (max(high) - min(low)))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(15)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # Volume confirmation (20-period average)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        # Only trade in ranging markets (Chop > 61.8) for mean reversion
        # In trending markets (Chop < 38.2), we could trend follow, but keep simple: only mean reversion in chop
        if chop[i] > 61.8:
            if position == 0:
                # Long when RSI oversold and price above KAMA (bullish bias in range)
                if rsi[i] < 30 and price > kama[i] and vol > 1.5 * vol_ma:
                    signals[i] = 0.25
                    position = 1
                # Short when RSI overbought and price below KAMA (bearish bias in range)
                elif rsi[i] > 70 and price < kama[i] and vol > 1.5 * vol_ma:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                # Exit long when RSI returns to neutral or price crosses below KAMA
                if rsi[i] > 50 or price < kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:
                # Exit short when RSI returns to neutral or price crosses above KAMA
                if rsi[i] < 50 or price > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In trending markets, stay flat to avoid whipsaws
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop_MeanReversion"
timeframe = "1d"
leverage = 1.0