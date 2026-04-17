#!/usr/bin/env python3
"""
Hypothesis: On the 4-hour timeframe, we combine the 12-hour Supertrend trend filter with
1-day volume-weighted average price (VWAP) as dynamic support/resistance, entering on
pullbacks to VWAP in the direction of the 12h Supertrend. Exits occur on opposite
Supertrend flip or when price extends 2 ATR away from VWAP (profit target). This
captures trend continuation moves while avoiding counter-trend noise, designed to
work in both bull (strong trends) and bear (sharp declines) markets with controlled
trade frequency (~20-40 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12h Supertrend (ATR=10, mult=3.0) for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range and ATR(10)
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper/Lower Bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + 3.0 * atr_10
    lower_band = hl2 - 3.0 * atr_10
    
    # Supertrend logic
    supertrend = np.full_like(close_12h, np.nan, dtype=float)
    dir_ = np.full_like(close_12h, 1, dtype=int)  # 1=up, -1=down
    
    for i in range(1, len(close_12h)):
        if np.isnan(atr_10[i-1]) or np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
            continue
        if close_12h[i] > upper_band[i-1]:
            dir_[i] = 1
        elif close_12h[i] < lower_band[i-1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i-1]
            if dir_[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if dir_[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        supertrend[i] = lower_band[i] if dir_[i] == 1 else upper_band[i]
    
    # Align Supertrend direction to 4h (wait for 12h bar close)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, dir_.astype(float))
    
    # === 1d VWAP as dynamic support/resistance ===
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Align VWAP to 4h
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === 4h ATR(14) for profit target/exit ===
    tr_4h1 = high[1:] - low[1:]
    tr_4h2 = np.abs(high[1:] - close[:-1])
    tr_4h3 = np.abs(low[1:] - close[:-1])
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation: 20-period volume MA ===
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = max(100, 20)  # warmup for Supertrend, VWAP, ATR, volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(vwap_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_aligned[i]
        atr = atr_14[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        trend = supertrend_dir_aligned[i]  # 1=up, -1=down
        
        if position == 0:
            # Look for pullback to VWAP in direction of 12h trend
            if trend == 1:  # uptrend
                # Long: price pulls back to near VWAP (within 0.5*ATR) with volume confirmation
                if abs(price - vwap) <= 0.5 * atr and vol > 1.5 * vol_ma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            else:  # downtrend
                # Short: price pulls back to near VWAP (within 0.5*ATR) with volume confirmation
                if abs(price - vwap) <= 0.5 * atr and vol > 1.5 * vol_ma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:  # long
            # Exit: opposite trend flip OR price extends 2*ATR above VWAP (profit target)
            if trend == -1 or price >= vwap + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # short
            # Exit: opposite trend flip OR price extends 2*ATR below VWAP (profit target)
            if trend == 1 or price <= vwap - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Supertrend12h_VWAP_Pullback"
timeframe = "4h"
leverage = 1.0