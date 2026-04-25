#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + ATR Regime Filter
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume confirmation ensures breakout legitimacy.
ATR-based regime filter avoids whipsaw in low-volatility ranging markets. Designed for 4h timeframe with 1d HTF trend filter.
Target: 20-50 trades/year per symbol to minimize fee drag while capturing explosive moves in both bull and bear markets.
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
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period ATR for volatility regime filter (4h)
    atr = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(20, n):
        atr[i] = np.mean(tr[i-19:i+1])
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate Donchian channels (20-period high/low)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, ATR, volume MA, and EMA34
    start_idx = max(34, 20)  # 34 for EMA34, 20 for Donchian/ATR/volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        ema_34_val = ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        # ATR-based regime filter: avoid trading in extremely low volatility (chop)
        # Use ATR ratio: current ATR / 50-period average ATR
        if i >= 50:
            atr_ma_50 = np.mean(atr[i-49:i+1])
            atr_ratio = atr_val / atr_ma_50 if atr_ma_50 > 0 else 0
            # Only trade when volatility is not extremely low (avoid chop)
            vol_regime_ok = atr_ratio > 0.7
        else:
            vol_regime_ok = True  # Not enough data for ATR MA, allow trading
        
        # Trend filter: price above/below 1d EMA34
        uptrend_filter = curr_close > ema_34_val
        downtrend_filter = curr_close < ema_34_val
        
        if position == 0:
            # Look for breakouts with volume confirmation and proper regime
            bull_breakout = (curr_close > donchian_high[i]) and volume_confirm and vol_regime_ok and uptrend_filter
            bear_breakout = (curr_close < donchian_low[i]) and volume_confirm and vol_regime_ok and downtrend_filter
            
            if bull_breakout:
                signals[i] = 0.25
                position = 1
            elif bear_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below Donchian low OR ATR spikes significantly (volatility expansion exit)
            if curr_close < donchian_low[i] or (i >= 50 and atr_val > 3.0 * np.mean(atr[i-49:i+1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR ATR spikes significantly
            if curr_close > donchian_high[i] or (i >= 50 and atr_val > 3.0 * np.mean(atr[i-49:i+1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ATRRegime"
timeframe = "4h"
leverage = 1.0