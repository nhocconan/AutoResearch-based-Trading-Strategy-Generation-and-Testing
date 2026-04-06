#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter with 1-day Donchian(20) breakout and volume confirmation.
# Uses daily trend direction to filter counter-trend trades, Choppiness Index to identify trending vs ranging markets,
# and volume to confirm breakouts. Targets 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
# Works in bull/bear by only trading breakouts in trending markets (CHOP < 38.2) with daily trend alignment.

name = "12h_chop1d_donchian20_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # 1-day Choppiness Index (14-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr_1d = np.zeros(len(close_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # ATR(14) for 1d
    atr_1d = np.zeros(len(close_1d))
    if len(close_1d) >= 14:
        atr_1d[13] = np.mean(tr_1d[:14])
        for i in range(14, len(close_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Sum of TR over 14 periods
    sum_tr_14 = np.zeros(len(close_1d))
    if len(close_1d) >= 14:
        sum_tr_14[13] = np.sum(tr_1d[:14])
        for i in range(14, len(close_1d)):
            sum_tr_14[i] = sum_tr_14[i-1] - tr_1d[i-14] + tr_1d[i]
    
    # Choppiness Index
    chop = np.full(len(close_1d), 50.0)  # default to middle range
    if len(close_1d) >= 14:
        high_max = np.zeros(len(close_1d))
        low_min = np.zeros(len(close_1d))
        for i in range(len(close_1d)):
            if i < 13:
                high_max[i] = np.max(high_1d[:i+1]) if i >= 0 else high_1d[0]
                low_min[i] = np.min(low_1d[:i+1]) if i >= 0 else low_1d[0]
            else:
                high_max[i] = np.max(high_1d[i-13:i+1])
                low_min[i] = np.min(low_1d[i-13:i+1])
        
        # Avoid division by zero
        range_14 = high_max - low_min
        chop = np.where(
            (sum_tr_14 > 0) & (range_14 > 0),
            100 * np.log10(sum_tr_14 / range_14) / np.log10(14),
            50.0
        )
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1-day EMA(50) for trend direction
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-day Donchian channels (20-period)
    donch_high_1d = np.full(len(close_1d), np.nan)
    donch_low_1d = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        donch_high_1d[i] = np.max(high_1d[i-20:i])
        donch_low_1d[i] = np.min(low_1d[i-20:i])
    
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(donch_high_1d_aligned[i]) or np.isnan(donch_low_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: Choppiness Index < 38.2 indicates trending market
        trending_market = chop_aligned[i] < 38.2
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below daily EMA or stoploss hit
            if (close[i] < ema_1d_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above daily EMA or stoploss hit
            if (close[i] > ema_1d_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries only in trending markets
            if trending_market and volume_filter:
                # Long: price breaks above daily Donchian high and above daily EMA (bullish)
                if (close[i] > donch_high_1d_aligned[i] and 
                    close[i] > ema_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below daily Donchian low and below daily EMA (bearish)
                elif (close[i] < donch_low_1d_aligned[i] and 
                      close[i] < ema_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals