#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA + RSI + Choppiness regime filter
# Long when KAMA rising and RSI > 50 in trending regime (Choppiness < 38.2)
# Short when KAMA falling and RSI < 50 in trending regime
# Uses 1d Choppiness Index to filter choppy markets, avoids whipsaws
# Target: 15-30 trades/year by requiring strong trend + momentum alignment
# Works in bull/bear: Choppiness filter ensures only trending markets are traded

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR (14-period Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    
    # Sum of True Range over 14 periods
    tr_sum = np.convolve(tr, np.ones(14), 'valid')
    tr_sum_padded = np.full(n, np.nan)
    tr_sum_padded[13:13+len(tr_sum)] = tr_sum
    
    # Highest high and lowest low over 14 periods
    max_high = np.maximum.accumulate(high_1d)
    min_low = np.minimum.accumulate(low_1d)
    range_max = np.maximum.accumulate(high_1d) - np.minimum.accumulate(low_1d)
    
    # Choppiness Index = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    chop = 100 * np.log10(tr_sum_padded / range_max) / np.log10(14)
    chop = np.where((range_max == 0) | np.isnan(tr_sum_padded), 100, chop)
    
    # Align Choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate KAMA on 12h data
    close = prices['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if False else None  # placeholder
    
    # Correct ER calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            price_volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
            if price_volatility > 0:
                er[i] = price_change / price_volatility
            else:
                er[i] = 0
    
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period) on 12h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(chop_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama[i] > kama[i-1] if i > 0 else False
        kama_falling = kama[i] < kama[i-1] if i > 0 else False
        
        # RSI condition
        rsi_above_50 = rsi[i] > 50
        rsi_below_50 = rsi[i] < 50
        
        # Regime filter: trending when Choppiness < 38.2
        trending_regime = chop_aligned[i] < 38.2
        
        if position == 0:
            if trending_regime:
                # Long: KAMA rising and RSI > 50
                if kama_rising and rsi_above_50:
                    signals[i] = 0.25
                    position = 1
                # Short: KAMA falling and RSI < 50
                elif kama_falling and rsi_below_50:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if KAMA turns down or RSI < 50 or regime becomes choppy
                if not kama_rising or not rsi_above_50 or not trending_regime:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if KAMA turns up or RSI > 50 or regime becomes choppy
                if not kama_falling or not rsi_below_50 or not trending_regime:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_KAMA_RSI_ChopRegime"
timeframe = "12h"
leverage = 1.0