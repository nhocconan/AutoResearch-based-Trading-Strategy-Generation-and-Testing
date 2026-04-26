#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h timeframe strategy using 1d Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume confirmation (>1.8x 20-bar avg volume).
Designed for low trade frequency (12-37/year) with discrete sizing (0.30) to minimize fee drag.
Works in both bull/bear: trend filter adapts to market direction, volume confirms institutional interest.
Only 12h bars considered for entries/exits to reduce overtrading and improve generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous 1d bar's high, low, close for Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R1, S1
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.0/12
    S1 = prev_close - camarilla_range * 1.0/12
    
    # Align Camarilla levels to 12h timeframe (1d -> 12h)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 1.8x average volume (using 12h volume)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop (14-period) - using 12h data
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 1d EMA (34), volume MA (20), ATR (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above R1, uptrend (close > 1d EMA34), volume spike
            long_signal = (high_val > R1_val) and (close_val > ema_34_1d_val) and (volume_val > 1.8 * vol_ma_val)
            # Short: break below S1, downtrend (close < 1d EMA34), volume spike
            short_signal = (low_val < S1_val) and (close_val < ema_34_1d_val) and (volume_val > 1.8 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_val)
            # Exit: trailing stop hit or trend reversal (price < 1d EMA34)
            if (low_val < long_stop) or (close_val < ema_34_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_val)
            # Exit: trailing stop hit or trend reversal (price > 1d EMA34)
            if (high_val > short_stop) or (close_val > ema_34_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0