#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA(14,2,30) direction + RSI(14) extreme + Chop(14) regime filter
# KAMA adapts to trend: stays flat in range, follows in trend
# RSI < 30 or > 70 identifies overextended moves for mean reversion
# Chop > 61.8 indicates ranging market where mean reversion works best
# Position size 0.25 to manage drawdown in choppy/trending markets
# Works in bull/bear as it fades extremes in ranging conditions
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA on close
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum() - np.abs(np.diff(close, prepend=close[0])).cumsum()
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_val = kama(close, 10, 2, 30)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl != 0, 100 * np.log10(sum_atr / range_hl) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(30, 14, 14)
    
    for i in range(start, n):
        if (np.isnan(kama_val[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama_val[i]
        price_below_kama = close[i] < kama_val[i]
        
        # RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Chop regime: ranging market
        chop_range = chop[i] > 61.8
        
        if position == 0:
            # Enter long: price below KAMA (dip) + RSI oversold + choppy market
            if price_below_kama and rsi_oversold and chop_range:
                position = 1
                signals[i] = position_size
            # Enter short: price above KAMA (rally) + RSI overbought + choppy market
            elif price_above_kama and rsi_overbought and chop_range:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above KAMA or RSI overbought
            if price_above_kama or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below KAMA or RSI oversold
            if price_below_kama or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMA_RSI_Chop_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0