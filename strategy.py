#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Uses daily timeframe with Camarilla R1/S1 breakouts filtered by 1-week EMA50 trend, volume confirmation, and ATR-based dynamic position sizing. Designed for BTC/ETH to work in both bull and bear markets by only taking breakouts in the direction of the 1w trend. Target ~15-25 trades/year to minimize fee drag.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 1d data for Camarilla levels (from previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 1d
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    # ATR for dynamic position sizing (based on 14-period ATR)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 1w EMA50 (50), 1d shift(1) for Camarilla, vol avg (20), ATR (14)
    start_idx = max(50 + 1, 1 + 1, 20, 14)  # ~51 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        atr_val = atr[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1w EMA50 alignment and volume confirmation
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_conf)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_conf)
            
            if long_condition:
                # Dynamic size based on ATR (normalized to 0.25-0.35 range)
                atr_norm = min(max(atr_val / close_val, 0.01), 0.05)  # cap between 1%-5%
                size = 0.25 + (atr_norm - 0.01) * (0.10 / 0.04)  # scale 0.01-0.05 to 0.25-0.35
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                atr_norm = min(max(atr_val / close_val, 0.01), 0.05)
                size = 0.25 + (atr_norm - 0.01) * (0.10 / 0.04)
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below 1w EMA50 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 1w EMA50 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0