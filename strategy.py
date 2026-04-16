#!/usr/bin/env python3
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
    
    # === Daily OHLC for ATR and close ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === ATR(14) calculation ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Daily True Range Percentile (20-day) ===
    tr_pct = pd.Series(tr).rolling(window=20, min_periods=20).rank(pct=True).values
    
    # === ATR Percentile aligned to 4h ===
    atr_pct_4h = align_htf_to_ltf(prices, df_1d, tr_pct)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === Price position relative to daily range ===
    # Calculate daily range position: (close - low) / (high - low)
    daily_range_pos = (close_1d - low_1d) / (high_1d - low_1d + 1e-10)
    # Align to 4h
    daily_range_pos_4h = align_htf_to_ltf(prices, df_1d, daily_range_pos)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_pct_4h[i]) or np.isnan(daily_range_pos_4h[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr_pct = atr_pct_4h[i]
        range_pos = daily_range_pos_4h[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Exit when volatility drops or range extreme reached ===
        if position == 1:  # Long position
            # Exit when volatility drops significantly or price reaches upper range
            if atr_pct < 0.3 or range_pos > 0.9:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when volatility drops significantly or price reaches lower range
            if atr_pct < 0.3 or range_pos < 0.1:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Low volatility + price in lower range + volume spike
            if atr_pct > 0.7 and range_pos < 0.3 and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Low volatility + price in upper range + volume spike
            elif atr_pct > 0.7 and range_pos > 0.7 and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_ATR_Percentile_Range_Position_Volume"
timeframe = "4h"
leverage = 1.0