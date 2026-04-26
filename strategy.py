#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_ChopFilter_V2
Hypothesis: On 12h timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI(14) for momentum confirmation, and Choppiness Index(14) as regime filter. Only take longs when KAMA rising, RSI>50, and CHOP<61.8 (trending regime); shorts when KAMA falling, RSI<50, and CHOP<61.8. Avoids whipsaws in ranging markets (CHOP>61.8) and reduces false signals. Discrete position sizing (0.25) minimizes fee churn. Designed to work in both bull and bear markets by adapting to trend strength and regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need warmup for KAMA, RSI, and CHOP
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === KAMA (Kaufman Adaptive Moving Average) on 12h close ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # Sum of |close[t] - close[t-1]| over 10 periods
    # Fix dimensions: change length is n-10, volatility length is n-1
    # We'll compute ER using a loop for correctness
    er = np.zeros(n)
    for i in range(10, n):
        price_change = np.abs(close[i] - close[i-10])
        sum_abs_diff = 0.0
        for j in range(1, 11):
            sum_abs_diff += np.abs(close[i-j+1] - close[i-j])
        if sum_abs_diff > 0:
            er[i] = price_change / sum_abs_diff
        else:
            er[i] = 0.0
    # Smoothing constants: fastest SC=2/(2+1)=0.6667, slowest SC=2/(30+1)=0.0645
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start at index 9 (first ER at index 10 uses up to index 9)
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) on 12h ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element (since diff reduces length by 1)
    rsi = np.concatenate([[np.nan], rsi])
    
    # === Choppiness Index(14) on 12h ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum_tr / (max_high - min_low)) / log10(14)
    range_hl = max_high - min_low
    # Avoid division by zero
    chop = np.where(range_hl > 0, 100 * np.log10(sum_tr / range_hl) / np.log10(14), 50)
    
    # === Volume confirmation (20-period SMA) ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    atr_multiplier = 2.0  # ATR stoploss multiplier
    
    # Calculate ATR(14) for stoploss
    tr_atr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_atr[0] = high[0] - low[0]
    atr = pd.Series(tr_atr).rolling(window=14, min_periods=14).mean().values
    
    # Start after warmup (need 20 for volume, 14 for RSI/CHOP/ATR, 10 for KAMA)
    start_idx = max(20, 14, 10)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_val = ema_34_1d_aligned[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or 
            np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(atr_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirmed = vol > 1.3 * avg_vol
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama_val > kama[i-1] if i > 0 else False
        kama_falling = kama_val < kama[i-1] if i > 0 else False
        
        # Long logic: KAMA rising, RSI>50, CHOP<61.8 (trending), volume confirmed
        long_condition = kama_rising and (rsi_val > 50) and (chop_val < 61.8) and volume_confirmed
        # Short logic: KAMA falling, RSI<50, CHOP<61.8 (trending), volume confirmed
        short_condition = kama_falling and (rsi_val < 50) and (chop_val < 61.8) and volume_confirmed
        
        # Exit logic: opposite KAMA direction or CHOP>61.8 (ranging regime)
        exit_long = (not kama_rising) or (chop_val >= 61.8)
        exit_short = (not kama_falling) or (chop_val >= 61.8)
        
        # ATR-based stoploss
        if position == 1:
            stop_price = entry_price - atr_multiplier * atr_val
            if close_val < stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            stop_price = entry_price + atr_multiplier * atr_val
            if close_val > stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_KAMA_Trend_RSI_ChopFilter_V2"
timeframe = "12h"
leverage = 1.0