#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA trend + RSI + chop regime (CHOP > 61.8 = range) for mean reversion
    # KAMA adapts to market noise - efficient in trending, avoids whipsaws in chop
    # RSI(14) for overbought/oversold signals in ranging markets
    # Chop filter ensures we only mean revert in ranging conditions (avoid trending markets)
    # Works in bull/bear: mean reversion in range, trend following when KAMA breaks out
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation (adaptive moving average)
    def calculate_kama(close_prices, er_length=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        ir = np.zeros_like(change)
        for i in range(er_length, len(close_prices)):
            direction = np.abs(close_prices[i] - close_prices[i-er_length])
            volatility = np.sum(change[i-er_length+1:i+1])
            ir[i] = direction / volatility if volatility != 0 else 0
        
        # Smoothing constants
        sc = (ir * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close_prices)
        kama[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    # RSI calculation
    def calculate_rsi(close_prices, length=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        
        for i in range(1, len(gain)):
            if i < length:
                avg_gain[i] = np.mean(gain[1:i+1]) if i > 0 else 0
                avg_loss[i] = np.mean(loss[1:i+1]) if i > 0 else 0
            else:
                avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
                avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Choppiness Index calculation
    def calculate_chop(high_prices, low_prices, close_prices, length=14):
        # True Range
        tr1 = high_prices - low_prices
        tr2 = np.abs(high_prices - np.roll(close_prices, 1))
        tr3 = np.abs(low_prices - np.roll(close_prices, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_prices[0] - low_prices[0]
        
        # Sum of True Ranges
        atr = np.zeros_like(tr)
        for i in range(length, len(tr)):
            atr[i] = np.sum(tr[i-length+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(high_prices)
        ll = np.zeros_like(low_prices)
        for i in range(length, len(high_prices)):
            hh[i] = np.max(high_prices[i-length+1:i+1])
            ll[i] = np.min(low_prices[i-length+1:i+1])
        
        # Choppiness Index
        chop = np.zeros_like(close_prices)
        for i in range(length, len(close_prices)):
            if hh[i] != ll[i]:
                chop[i] = 100 * np.log10(atr[i] / (hh[i] - ll[i])) / np.log10(length)
            else:
                chop[i] = 50  # neutral when no range
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    rsi = calculate_rsi(close, length=14)
    chop = calculate_chop(high, low, close, length=14)
    
    # Load 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for trend filter
    kama_1d = calculate_kama(df_1d['close'].values, er_length=10, fast_sc=2, slow_sc=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_above_average = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(kama_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Market regime: chop > 61.8 indicates ranging market (good for mean reversion)
        is_ranging = chop[i] > 61.8
        
        if position == 0:
            # Long conditions: oversold RSI in ranging market OR KAMA breakout in trending
            if (is_ranging and rsi[i] < 30) or (not is_ranging and close[i] > kama_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: overbought RSI in ranging market OR KAMA breakdown in trending
            elif (is_ranging and rsi[i] > 70) or (not is_ranging and close[i] < kama_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:  # Long position
                # Exit on overbought RSI in range OR KAMA breakdown
                if (is_ranging and rsi[i] > 70) or (not is_ranging and close[i] < kama_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Short position
                # Exit on oversold RSI in range OR KAMA breakout
                if (is_ranging and rsi[i] < 30) or (not is_ranging and close[i] > kama_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0