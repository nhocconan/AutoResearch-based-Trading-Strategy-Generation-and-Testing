#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d KAMA + RSI + Chop Filter
# Hypothesis: KAMA adapts to market regime, capturing trends in both bull and bear markets.
# RSI filters for momentum strength, and Choppiness Index avoids whipsaws in ranging markets.
# Uses weekly trend filter for alignment with higher timeframe momentum.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    def kama(close, er_period=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, n=10))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[10:] = change[10:] / volatility[10:]
        er[np.isnan(er)] = 0
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_val = kama(close)
    kama_dir = np.where(kama_val > np.roll(kama_val, 1), 1, -1)
    kama_dir[0] = 1  # Initialize
    
    # Calculate RSI(14)
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_val = rsi(close)
    
    # Calculate Choppiness Index (14)
    def chop(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        max_high = np.zeros_like(high)
        min_low = np.zeros_like(low)
        max_high[0] = high[0]
        min_low[0] = low[0]
        for i in range(1, len(close)):
            max_high[i] = max(max_high[i-1], high[i])
            min_low[i] = min(min_low[i-1], low[i])
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if atr[i] > 0:
                chop[i] = 100 * np.log10((atr[i] * period) / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    chop_val = chop(high, low, close)
    
    # Weekly EMA(20) for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or 
            np.isnan(chop_val[i]) or np.isnan(weekly_ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR RSI overbought OR chop becomes too high (trending)
            if (kama_dir[i] == -1 or rsi_val[i] > 70 or chop_val[i] < 38.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR RSI oversold OR chop becomes too high (trending)
            if (kama_dir[i] == 1 or rsi_val[i] < 30 or chop_val[i] < 38.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Choppy market (chop > 61.8) for mean reversion
            if chop_val[i] > 61.8:
                # Long: price below KAMA AND RSI oversold
                if close[i] < kama_val[i] and rsi_val[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Short: price above KAMA AND RSI overbought
                elif close[i] > kama_val[i] and rsi_val[i] > 70:
                    position = -1
                    signals[i] = -0.25
            # Trending market (chop < 38.2) - follow weekly trend
            elif chop_val[i] < 38.2:
                # Only take longs in uptrend, shorts in downtrend
                if close[i] > weekly_ema_aligned[i]:  # Uptrend
                    if close[i] > kama_val[i] and rsi_val[i] > 50:
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if close[i] < kama_val[i] and rsi_val[i] < 50:
                        position = -1
                        signals[i] = -0.25
    
    return signals