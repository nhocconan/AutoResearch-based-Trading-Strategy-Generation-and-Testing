#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter
Hypothesis: Daily KAMA direction with RSI overbought/oversold and Choppiness Index regime filter.
KAMA adapts to market noise, reducing whipsaw in chop. RSI extremes provide mean-reversion entries.
Chop filter (>61.8) ensures we only mean-revert in ranging markets, avoiding trending whipsaw.
Targets 15-25 trades/year by requiring KAMA trend, RSI extreme, and chop regime alignment.
Works in bull/bear: mean reversion in ranges, trend following via KAMA in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, k=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle first er_len values
    er = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(er_len, len(close)):
        if volatility[i] != 0:
            er[i] = change[i-er_len] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]  # Seed
    for i in range(er_len + 1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    # First average
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Wilder smoothing
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.full_like(close, np.nan)
    for i in range(1, len(close)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr14 = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        atr14[i] = np.mean(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh = np.full_like(close, np.nan)
    ll = np.full_like(close, np.nan)
    for i in range(13, len(close)):
        hh[i] = np.max(high[i-13:i+1])
        ll[i] = np.min(low[i-13:i+1])
    
    chop = np.full_like(close, np.nan)
    for i in range(13, len(close)):
        if atr14[i] > 0 and hh[i] > ll[i]:
            chop[i] = 100 * np.log10(sum(tr[i-13:i+1]) / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    # Align KAMA, RSI, Chop to 1d (already 1d but ensure alignment)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Chop > 61.8 = ranging market (mean revert)
        ranging = chop_aligned[i] > 61.8
        
        # KAMA trend: price above KAMA = bullish bias, below = bearish bias
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Entry logic: only in ranging markets
        if ranging:
            # Long: RSI oversold + price above KAMA (bullish bias in range)
            if rsi_oversold and price_above_kama and position <= 0:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + price below KAMA (bearish bias in range)
            elif rsi_overbought and price_below_kama and position >= 0:
                signals[i] = -0.25
                position = -1
            # Exit when RSI returns to neutral (40-60) or opposite extreme
            elif position == 1 and (rsi_aligned[i] > 50 or rsi_overbought):
                signals[i] = -0.25  # Close long
                position = 0
            elif position == -1 and (rsi_aligned[i] < 50 or rsi_oversold):
                signals[i] = 0.25   # Close short
                position = 0
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # In trending markets, follow KAMA direction
            if price_above_kama and position <= 0:
                signals[i] = 0.25
                position = 1
            elif price_below_kama and position >= 0:
                signals[i] = -0.25
                position = -1
            elif position == 1 and price_below_kama:
                signals[i] = 0.0  # Exit long
                position = 0
            elif position == -1 and price_above_kama:
                signals[i] = 0.0  # Exit short
                position = 0
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0