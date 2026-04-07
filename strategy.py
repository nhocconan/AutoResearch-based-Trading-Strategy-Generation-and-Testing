#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d KAMA + RSI + Chop Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
# RSI filters for overbought/oversold conditions, while Choppiness Index identifies ranging vs trending regimes.
# In trending markets (CHOP < 38.2), follow KAMA direction. In ranging markets (CHOP > 61.8), fade RSI extremes.
# Volume confirmation ensures institutional participation. Designed for low trade frequency to minimize drag.

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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # KAMA (Adaptive Moving Average) - 10-period ER, 2/30 SC
    close_s = pd.Series(close)
    change = abs(close_s.diff(10)).values
    volatility = abs(close_s.diff(1)).rolling(window=10, min_periods=1).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.zeros(n)
    for i in range(14, n):
        if highest_high[i] > lowest_low[i]:
            sum_atr = np.sum(atr[i-13:i+1])
            chop[i] = 100 * np.log10(sum_atr / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral if no range
    
    # Weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_ok = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        if not vol_ok[i]:
            signals[i] = 0.0
            continue
        
        # Regime filter: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        if chop[i] < 38.2:  # Trending regime
            if position == 1:  # Long position
                # Exit: price below KAMA or weekly trend turns bearish
                if close[i] < kama[i] or close[i] < ema_20_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Exit: price above KAMA or weekly trend turns bullish
                if close[i] > kama[i] or close[i] > ema_20_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Flat, look for entry
                # Enter long: price above KAMA and weekly uptrend
                if close[i] > kama[i] and close[i] > ema_20_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: price below KAMA and weekly downtrend
                elif close[i] < kama[i] and close[i] < ema_20_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
        else:  # Ranging regime (CHOP > 61.8) or neutral
            # Mean reversion: fade RSI extremes
            if position == 1:  # Long position
                # Exit: RSI overbought or mean reversion signal
                if rsi[i] > 70:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Exit: RSI oversold or mean reversion signal
                if rsi[i] < 30:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Flat, look for entry
                # Enter long: RSI oversold (<30)
                if rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Enter short: RSI overbought (>70)
                elif rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals