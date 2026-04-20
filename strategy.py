#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_Regime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Close for price action ===
    close_1d = df_1d['close'].values
    
    # === Weekly Candles for Context ===
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # === Weekly Trend Filter (EMA34) ===
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Daily Previous Values for Pivot Calculation ===
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(close_1d, 1)
    
    # === Daily Camarilla Pivot Points ===
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # === Align Weekly EMA to Daily ===
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === Align Daily Pivot Levels to Daily (no shift needed, but for consistency) ===
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === Daily Volume Confirmation ===
    volume = df_1d['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.full_like(volume, np.nan), where=vol_ma20!=0)
    
    # === Daily Chopiness Index (14) for Regime Filter ===
    high_low = df_1d['high'].values - df_1d['low'].values
    atr1 = np.zeros_like(high_low)
    atr1[0] = high_low[0]
    for i in range(1, len(high_low)):
        tr = max(
            high_low[i],
            abs(df_1d['high'].values[i] - df_1d['close'].values[i-1]),
            abs(df_1d['low'].values[i] - df_1d['close'].values[i-1])
        )
        atr1[i] = 0.9 * atr1[i-1] + 0.1 * tr
    
    # Sum of absolute returns over 14 days
    abs_returns = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    sum_abs_ret = np.zeros_like(abs_returns)
    for i in range(len(abs_returns)):
        if i < 14:
            sum_abs_ret[i] = np.sum(abs_returns[:i+1])
        else:
            sum_abs_ret[i] = np.sum(abs_returns[i-13:i+1])
    
    # Chopiness index formula
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if sum_abs_ret[i] > 0 and atr1[i] > 0:
            chop[i] = 100 * np.log10(sum_abs_ret[i] / (atr1[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Align chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close_1d[i]
        vol_ratio_val = vol_ratio[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        chop_val = chop_aligned[i]
        weekly_ema_val = weekly_ema_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(pivot_val) or np.isnan(chop_val) or 
            np.isnan(weekly_ema_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume confirmation and weekly uptrend in low chop
            if close_val > r1_val and vol_ratio_val > 2.0 and weekly_ema_val < close_val and chop_val < 50:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume confirmation and weekly downtrend in low chop
            elif close_val < s1_val and vol_ratio_val > 2.0 and weekly_ema_val > close_val and chop_val < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below pivot OR chop increases (rangy market) OR weekly trend turns down
            if close_val < pivot_val or chop_val > 60 or weekly_ema_val > close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above pivot OR chop increases OR weekly trend turns up
            if close_val > pivot_val or chop_val > 60 or weekly_ema_val < close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals