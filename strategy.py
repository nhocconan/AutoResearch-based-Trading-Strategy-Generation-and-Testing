#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_Spike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly Trend Filter (EMA34) ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Data for Pivot Calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Classic pivot (same for Camarilla)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels - Focus on R1/S1 for breakouts
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume Spike Detection (20-bar z-score) ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_mean = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_std = vol_series.rolling(window=20, min_periods=20).std().values
    vol_z = np.where(vol_std > 0, (volume - vol_mean) / vol_std, 0)
    
    # === ATR for Stop Loss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_series = pd.Series(tr)
    atr = atr_series.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema_34_val = ema_34_1w_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_z_val = vol_z[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_34_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(vol_z_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout above R1 with weekly uptrend and volume spike
            if close_val > r1_val and close_val > ema_34_val and vol_z_val > 2.0:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short breakout below S1 with weekly downtrend and volume spike
            elif close_val < s1_val and close_val < ema_34_val and vol_z_val > 2.0:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: stop loss or price returns to S1
            if close_val <= entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            elif close_val < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or price returns to S1
            if close_val >= entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            elif close_val > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals