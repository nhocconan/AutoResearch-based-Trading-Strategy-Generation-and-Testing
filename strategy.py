#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout with 1d ATR-based trend filter and volume spike (>1.8x median) on 4h. Uses ATR trailing stop (2.5x) for risk control. Designed for BTC/ETH with moderate trade frequency (~30-40/year) to balance edge capture and fee drag. Uses tighter R1/S1 levels for more frequent but still selective entries in both bull and bear markets via volatility-adjusted trend filter.
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
    
    # Get 1d data for HTF trend (ATR-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ATR(14) for trend filter: trend up if close > close + 0.5*ATR, down if close < close - 0.5*ATR
    # Using 1d ATR to normalize the trend filter by volatility
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'], 
                       np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)), 
                                  np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d EMA(close) for trend direction reference
    ema_close_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Trend filter: uptrend if EMA20 rising and price above EMA20 + 0.3*ATR, downtrend if falling and below EMA20 - 0.3*ATR
    ema_close_1d_shift = np.roll(ema_close_1d, 1)
    ema_close_1d_shift[0] = ema_close_1d[0]
    ema_rising = ema_close_1d > ema_close_1d_shift
    ema_falling = ema_close_1d < ema_close_1d_shift
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar (HLC of prior 4h)
    cam_high = pd.Series(df_4h['high'].values).shift(1).values
    cam_low = pd.Series(df_4h['low'].values).shift(1).values
    cam_close = pd.Series(df_4h['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels (core breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # Volume spike filter: volume > 1.8x median volume (20-period) for conviction
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops on 4h
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_close_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_close_1d)
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling.astype(float))
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(20) 1d, Camarilla (need 2 bars for shift), volume median (20), ATR (14)
    start_idx = max(20, 2, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_close_1d_aligned[i]) or 
            np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_close_1d_val = ema_close_1d_aligned[i]
        ema_rising_val = bool(ema_rising_aligned[i])
        ema_falling_val = bool(ema_falling_aligned[i])
        atr_1d_val = atr_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        
        # Dynamic trend filter: uptrend if EMA rising AND price > EMA + 0.3*ATR(1d)
        # Downtrend if EMA falling AND price < EMA - 0.3*ATR(1d)
        uptrend = ema_rising_val and (close_val > ema_close_1d_val + 0.3 * atr_1d_val)
        downtrend = ema_falling_val and (close_val < ema_close_1d_val - 0.3 * atr_1d_val)
        
        # Volume spike filter: only trade in high-volume environments
        volume_spike = volume_val > 1.8 * vol_median_val
        
        if position == 0:
            # Long: break above R1 with volume spike, and uptrend
            long_signal = (close_val > r1_val) and \
                          volume_spike and \
                          uptrend
            
            # Short: break below S1 with volume spike, and downtrend
            short_signal = (close_val < s1_val) and \
                           volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop (2.5x ATR for slightly wider stop to avoid whipsaw)
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop (2.5x ATR)
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0