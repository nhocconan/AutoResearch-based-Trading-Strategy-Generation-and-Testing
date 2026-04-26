#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) as regime filter.
Only trade when KAMA slope aligns with RSI > 50 (long) or < 50 (short) AND market is
trending (CHOP < 38.2) or ranging (CHOP > 61.8) with appropriate logic.
In trending markets: follow KAMA direction. In ranging markets: mean revert at RSI extremes.
Designed for low trade frequency (7-25/year) to minimize fee drag while working in
both bull (2021-2024) and bear (2025+) markets via regime adaptation.
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
    
    # === 1W HTF for trend filter (optional, can be removed if too restrictive) ===
    # df_1w = get_htf_data(prices, '1w')
    # if len(df_1w) < 50:
    #     return np.zeros(n)
    # close_1w = df_1w['close'].values
    # ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    # weekly_trend = close_1w[-1] > ema_20_1w[-1] if len(close_1w) > 0 else True  # placeholder
    
    # === 1d indicators (no MTF needed as primary is 1d) ===
    # KAMA calculation
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if hasattr(np, 'sum') else np.abs(np.diff(close, n=1)).sum()
        # Fix: calculate properly
        change = np.abs(close[length:] - close[:-length])
        volatility = []
        for i in range(length, len(close)):
            volatility.append(np.sum(np.abs(np.diff(close[i-length+1:i+1]))))
        volatility = np.array(volatility)
        er = np.zeros_like(close)
        er[length:] = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.full_like(close, np.nan)
        kama[length] = close[length]
        for i in range(length+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI
    def calculate_rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(span=length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        # Prepend NaN for first element
        rsi = np.concatenate([[np.nan], rsi])
        return rsi
    
    # Choppiness Index
    def calculate_chop(high, low, close, length=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        # Sum of TR over period
        tr_sum = pd.Series(tr).rolling(window=length, min_periods=length).sum().values
        # Highest high and lowest low over period
        hh = pd.Series(high).rolling(window=length, min_periods=length).max().values
        ll = pd.Series(low).rolling(window=length, min_periods=length).min().values
        # Chop formula
        chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(length)
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    rsi = calculate_rsi(close, length=14)
    chop = calculate_chop(high, low, close, length=14)
    
    # KAMA slope (direction)
    kama_slope = np.diff(kama, prepend=np.nan)
    
    # Volume confirmation (optional)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)  # milder spike
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(kama_slope[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position
        
        # Regime determination
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        # Neutral zone (38.2 <= CHOP <= 61.8) - no trade
        
        if position == 0:
            # Look for entry
            if is_trending:
                # In trending market: follow KAMA slope with RSI confirmation
                long_entry = (kama_slope[i] > 0) and (rsi[i] > 50) and volume_spike[i]
                short_entry = (kama_slope[i] < 0) and (rsi[i] < 50) and volume_spike[i]
            elif is_ranging:
                # In ranging market: mean revert at RSI extremes
                long_entry = (rsi[i] < 30) and volume_spike[i]  # oversold
                short_entry = (rsi[i] > 70) and volume_spike[i]  # overbought
            else:
                long_entry = False
                short_entry = False
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit conditions
            exit = False
            if is_trending:
                # Exit when KAMA slope turns negative or RSI < 40
                if kama_slope[i] < 0 or rsi[i] < 40:
                    exit = True
            elif is_ranging:
                # Exit when RSI reaches 50 (mean) or adverse move
                if rsi[i] > 50 or rsi[i] < 20:  # took profit or stopped
                    exit = True
            # Always exit if volume dries up (optional)
            if not volume_spike[i] and i > start_idx + 5:  # allow some time
                exit = exit or True  # milder: exit on low volume after min hold
            
            if exit:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        
        elif position == -1:
            # Short exit conditions
            exit = False
            if is_trending:
                # Exit when KAMA slope turns positive or RSI > 60
                if kama_slope[i] > 0 or rsi[i] > 60:
                    exit = True
            elif is_ranging:
                # Exit when RSI reaches 50 (mean) or adverse move
                if rsi[i] < 50 or rsi[i] > 80:
                    exit = True
            # Always exit if volume dries up (optional)
            if not volume_spike[i] and i > start_idx + 5:
                exit = exit or True
            
            if exit:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0