#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_Regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Weekly Close for Trend Filter ===
    close_1w = df_1w['close'].values
    weekly_ema = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === Daily Close Price ===
    close_1d = df_1d['close'].values
    
    # === Daily ATR for Volatility Filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = np.maximum(high_1d - low_1d,
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]
    atr1 = np.zeros_like(tr1)
    atr1[0] = tr1[0]
    for i in range(1, len(tr1)):
        atr1[i] = 0.95 * atr1[i-1] + 0.05 * tr1[i]
    atr1_aligned = align_htf_to_ltf(prices, df_1d, atr1)
    
    # === Daily Volume Ratio ===
    volume_1d = df_1d['volume'].values
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # === Daily Chopiness Index (14) for Regime Filter ===
    # True Range
    tr = np.maximum(high_1d - low_1d,
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    # ATR
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = 0.95 * atr[i-1] + 0.05 * tr[i]
    # Sum of absolute returns over 14 periods
    abs_returns = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    sum_abs_ret = np.zeros_like(abs_returns)
    for i in range(len(abs_returns)):
        if i < 14:
            sum_abs_ret[i] = np.sum(abs_returns[:i+1])
        else:
            sum_abs_ret[i] = np.sum(abs_returns[i-13:i+1])
    # Chopiness index
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if sum_abs_ret[i] > 0 and atr[i] > 0:
            chop[i] = 100 * np.log10(sum_abs_ret[i] / (atr[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === Daily Previous Day's OHLC for Pivot Calculation ===
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first values to avoid look-ahead
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Classic Pivot Point (same as Camarilla base)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align all to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        weekly_ema_val = weekly_ema_aligned[i]
        atr_val = atr1_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        chop_val = chop_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(weekly_ema_val) or np.isnan(atr_val) or np.isnan(vol_ratio_val) or
            np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(pivot_val) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: only long in uptrend, short in downtrend
        weekly_uptrend = close_1d[i] > weekly_ema_val
        weekly_downtrend = close_1d[i] < weekly_ema_val
        
        if position == 0:
            # Long: Break above R1 with volume confirmation in trending market (low chop)
            if weekly_uptrend and close_val > r1_val and vol_ratio_val > 2.0 and chop_val < 50:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume confirmation in trending market (low chop)
            elif weekly_downtrend and close_val < s1_val and vol_ratio_val > 2.0 and chop_val < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below pivot OR chop increases significantly (rangy market)
            if close_val < pivot_val or chop_val > 65:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above pivot OR chop increases significantly
            if close_val > pivot_val or chop_val > 65:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals