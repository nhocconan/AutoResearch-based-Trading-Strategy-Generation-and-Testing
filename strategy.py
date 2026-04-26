#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter_VolumeSpike
Hypothesis: 12h Camarilla R1/S1 breakouts with 1w EMA trend filter and choppiness regime filter to avoid sideways markets. Volume spike confirmation ensures institutional participation. Designed for low trade frequency (<30/year) on 12h timeframe to minimize fee drag while capturing strong trending moves in both bull and bear markets via multi-timeframe alignment.
"""

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
    
    # Get 1w data for EMA trend filter (weekly timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA (34-period) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d ATR (14-period) for volatility measurement
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Previous 1d bar's high, low, close for Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R1, S1
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.0/12
    S1 = prev_close - camarilla_range * 1.0/12
    
    # Align HTF indicators to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 2.0x average volume (balanced for 12h frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop (14-period on 12h)
    tr1_12h = high[1:] - low[1:]
    tr2_12h = np.abs(high[1:] - close[:-1])
    tr3_12h = np.abs(low[1:] - close[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index regime filter (14-period on 12h) to avoid sideways markets
    # CHOP > 61.8 = ranging/choppy, CHOP < 38.2 = trending
    high_roll = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low_roll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_sum = pd.Series(atr_14_12h).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / np.log10(high_roll - low_roll)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((high_roll - low_roll) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 1w EMA (34), 1d ATR (14), volume MA (20), 12h ATR (14), chop (14)
    start_idx = max(34, 14, 20, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr_14_12h[i]) or 
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1w_val = ema_34_1w_aligned[i]
        atr_14_1d_val = atr_14_1d_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_14_12h_val = atr_14_12h[i]
        chop_val = chop[i]
        
        # Only trade in trending regimes (CHOP < 50.0) to avoid whipsaws in sideways markets
        trending_regime = chop_val < 50.0
        
        if position == 0:
            # Long: break above R1, above weekly EMA (uptrend), volume spike, trending regime
            long_signal = (high_val > R1_val) and (close_val > ema_34_1w_val) and \
                         (volume_val > 2.0 * vol_ma_val) and trending_regime
            # Short: break below S1, below weekly EMA (downtrend), volume spike, trending regime
            short_signal = (low_val < S1_val) and (close_val < ema_34_1w_val) and \
                          (volume_val > 2.0 * vol_ma_val) and trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.5 * atr_14_12h_val  # Wider stop for 12h
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.5 * atr_14_12h_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.5 * atr_14_12h_val)
            # Exit: trailing stop hit or trend reversal (close < weekly EMA) or chop regime
            if (low_val < long_stop) or (close_val < ema_34_1w_val) or (chop_val >= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.5 * atr_14_12h_val)
            # Exit: trailing stop hit or trend reversal (close > weekly EMA) or chop regime
            if (high_val > short_stop) or (close_val > ema_34_1w_val) or (chop_val >= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter_VolumeSpike"
timeframe = "12h"
leverage = 1.0