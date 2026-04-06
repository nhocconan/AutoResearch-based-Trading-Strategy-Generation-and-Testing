#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter combined with weekly EMA trend
# and daily volume confirmation. Uses weekly EMA for trend direction (works in bull/bear),
# Choppiness Index to detect ranging markets (avoid false breakouts), and volume
# spike to confirm genuine moves. Designed for 12h timeframe to target 50-150 trades
# over 4 years (12-37/year) with low frequency to minimize fee drag.

name = "12h_chop1w_ema_dailyvol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA(50) for trend direction - HTF
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 48) / 50
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily Choppiness Index(14) for regime detection - HTF
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.full(len(close_1d), np.nan)
    if len(close_1d) > 1:
        tr[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]),
                       abs(low_1d[i] - close_1d[i-1]))
    
    # ATR(14)
    atr_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        atr_1d[13] = np.mean(tr[1:14])
        for i in range(14, len(close_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    sum_atr_14 = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        sum_atr_14[i] = np.sum(atr_1d[i-13:i+1])
    
    # Max high and min low over 14 periods
    max_high_14 = np.full(len(close_1d), np.nan)
    min_low_14 = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        max_high_14[i] = np.max(high_1d[i-13:i+1])
        min_low_14[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index
    chop = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        if sum_atr_14[i] > 0 and max_high_14[i] > min_low_14[i]:
            chop[i] = 100 * np.log10(sum_atr_14[i] / (max_high_14[i] - min_low_14[i])) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Daily volume average - HTF
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(10, len(vol_1d)):  # 10-day average
        vol_ma_1d[i] = np.mean(vol_1d[i-10:i])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 13, 10)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Choppiness regime: avoid ranging markets (Chop > 61.8)
        ranging_market = chop_aligned[i] > 61.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend reversal or stoploss
            if (close[i] < ema_1w_aligned[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):  # simple range-based stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend reversal or stoploss
            if (close[i] > ema_1w_aligned[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries only in trending markets
            if not ranging_market and volume_filter:
                # Long: price above weekly EMA
                if close[i] > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price below weekly EMA
                elif close[i] < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals