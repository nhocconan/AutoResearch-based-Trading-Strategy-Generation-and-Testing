#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_ChopFilter_VolumeSpike
Hypothesis: Daily Camarilla breakout with weekly EMA50 trend filter, volume confirmation, and choppiness regime filter.
Works in bull/bear: Weekly EMA50 adapts to long-term trend direction, chop filter avoids whipsaws in ranging markets.
Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while preserving edge.
Uses discrete position sizing (0.25) to reduce fee churn. Only trades when price breaks Camarilla R1/S1 levels
with volume spike (>2.0x average) and favorable regime (CHOP < 55.0 = trending market).
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
    
    # Get 1d data for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d EMA14 for ATR calculation (using close prices)
    close_1d = df_1d['close'].values
    ema_14_1d = pd.Series(close_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Previous 1d bar's high, low, close for Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R1, S1
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.0/12
    S1 = prev_close - camarilla_range * 1.0/12
    
    # Align Camarilla levels to 1d timeframe (already aligned since we're using 1d data)
    # But we need to shift by 1 to avoid look-ahead (use previous day's levels for today's breakout)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR for stop (14-period using 1d data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume confirmation: 2.0x average volume (using 1d volume)
    vol_1d = df_1d['volume'].values
    vol_ma = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # Choppiness Index regime filter (14-period using 1d data)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_14 * 14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    chop_raw = np.where((max_high_14 - min_low_14) <= 0, 100, chop_raw)
    chop_raw = np.where(np.isnan(chop_raw) | np.isinf(chop_raw), 50, chop_raw)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 1d EMA (14), volume MA (20), ATR (14), CHOP (14), 1w EMA (50)
    start_idx = max(14, 20, 14, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma_aligned[i]
        atr_val = atr_aligned[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: only trade when not too choppy (CHOP < 55.0 = strong trending market)
        regime_filter = chop_val < 55.0
        
        if position == 0:
            # Long: break above R1, uptrend (close > 1w EMA50), volume spike, good regime
            long_signal = (high_val > R1_val) and (close_val > ema_50_1w_val) and (volume_val > 2.0 * vol_ma_val) and regime_filter
            # Short: break below S1, downtrend (close < 1w EMA50), volume spike, good regime
            short_signal = (low_val < S1_val) and (close_val < ema_50_1w_val) and (volume_val > 2.0 * vol_ma_val) and regime_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.5 * atr_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.5 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.5 * atr_val)
            # Exit: trailing stop hit or trend reversal (price < 1w EMA50) or regime becomes too choppy
            if (low_val < long_stop) or (close_val < ema_50_1w_val) or (chop_val >= 55.0):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.5 * atr_val)
            # Exit: trailing stop hit or trend reversal (price > 1w EMA50) or regime becomes too choppy
            if (high_val > short_stop) or (close_val > ema_50_1w_val) or (chop_val >= 55.0):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_ChopFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0