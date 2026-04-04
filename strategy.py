#!/usr/bin/env python3
"""
Experiment #4774: 1h Donchian(20) Breakout + 4h/1d Trend + Volume Spike
HYPOTHESIS: On 1h timeframe, Donchian(20) breakouts in direction of 4h HMA21 and 1d EMA200 trend with volume confirmation (>1.5x average) capture strong momentum moves. Uses 4h/1d for signal direction, 1h only for entry timing. Session filter (08-20 UTC) reduces noise trades. Fixed position size 0.20 to manage risk and minimize fee churn. Designed for 15-37 trades/year on 1h timeframe (60-150 total over 4 years) to avoid fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4774_1h_donchian20_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session filter (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Precompute HTF: 4h data for HMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    # Precompute HTF: 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: HMA21 for trend filter ===
    if len(df_4h) >= 21:
        # Hull Moving Average calculation
        half_len = len(df_4h) // 2
        sqrt_len = int(np.sqrt(len(df_4h)))
        
        # WMA function
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        close_4h = df_4h['close'].values
        wma_half = np.array([wma(close_4h[i:i+half_len], half_len)[-1] 
                            if i+half_len <= len(close_4h) else np.nan 
                            for i in range(len(close_4h))])
        wma_full = np.array([wma(close_4h[i:i+len(close_4h)], len(close_4h))[-1] 
                            if i+len(close_4h) <= len(close_4h) else np.nan 
                            for i in range(len(close_4h))])
        wma_sqrt = np.array([wma(close_4h[i:i+sqrt_len], sqrt_len)[-1] 
                            if i+sqrt_len <= len(close_4h) else np.nan 
                            for i in range(len(close_4h))])
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        hma_raw = 2 * wma_half - wma_full
        hma_4h = np.array([wma(hma_raw[i:i+sqrt_len], sqrt_len)[-1] 
                          if i+sqrt_len <= len(hma_raw) else np.nan 
                          for i in range(len(hma_raw))])
    else:
        hma_4h = np.full(len(df_4h), np.nan)
    
    # Align HTF HMA21 to 1h timeframe
    if len(hma_4h) > 0:
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    else:
        hma_4h_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: EMA200 for trend filter ===
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    else:
        ema_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF EMA200 to 1h timeframe
    if len(ema_1d) > 0:
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 200)  # Donchian, Volume MA, EMA200 warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(hma_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse on opposite signal ---
        if in_position:
            # Check for reverse signal
            vol_confirm = vol_ratio[i] > 1.5
            breakout_long = (price >= high_roll[i]) and (price > hma_4h_aligned[i]) and (price > ema_1d_aligned[i]) and vol_confirm
            breakout_short = (price <= low_roll[i]) and (price < hma_4h_aligned[i]) and (price < ema_1d_aligned[i]) and vol_confirm
            
            if position_side > 0 and breakout_short:
                # Long to short reversal
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            elif position_side < 0 and breakout_long:
                # Short to long reversal
                in_position = True
                position_side = 1
                signals[i] = SIZE
            else:
                # Hold current position
                signals[i] = SIZE * position_side
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with 4h/1d trend alignment
        breakout_long = (price >= high_roll[i]) and (price > hma_4h_aligned[i]) and (price > ema_1d_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < hma_4h_aligned[i]) and (price < ema_1d_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals