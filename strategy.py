#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation captures institutional breakouts in both bull and bear markets. The 1d EMA50 filter ensures alignment with higher timeframe trend, reducing false breakouts during counter-trend moves. Volume spike (>2.0x 20-period average) confirms participation. Designed for low trade frequency (~12-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily trend filter: 50-period EMA on 1d ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Donchian channels (20-period on 6h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian upper/lower bands (20-period lookback)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume spike filter (20-period on 6h) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.where(np.isnan(vol_ma) | (vol_ma == 0), 0, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        daily_ema = ema_50_1d_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume spike > 2.0 + price above daily EMA50 (bullish trend)
            if price_close > upper and vol_spike > 2.0 and price_close > daily_ema:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below Donchian lower + volume spike > 2.0 + price below daily EMA50 (bearish trend)
            elif price_close < lower and vol_spike > 2.0 and price_close < daily_ema:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
                lowest_since_entry = price_close
        
        elif position != 0:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price_high)
                # Time-based exit: exit after 3 bars (18 hours) to avoid overstaying
                # ATR-based trailing stop: 3.0 * ATR(14) below highest since entry
                atr_period = 14
                if i >= atr_period:
                    tr1 = high[i-atperiod:i+1] - low[i-atperiod:i+1]
                    tr2 = np.abs(high[i-atperiod:i+1] - np.roll(close[i-atperiod:i+1], 1))
                    tr3 = np.abs(low[i-atperiod:i+1] - np.roll(close[i-atperiod:i+1], 1))
                    tr = np.maximum(tr1, np.maximum(tr2, tr3))
                    tr[0] = high[i-atperiod] - low[i-atperiod] if i-atperiod >= 0 else tr1[0]
                    atr_val = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().iloc[-1]
                else:
                    atr_val = 0
                
                if price_close < highest_since_entry - 3.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                if i >= atr_period:
                    tr1 = high[i-atperiod:i+1] - low[i-atperiod:i+1]
                    tr2 = np.abs(high[i-atperiod:i+1] - np.roll(close[i-atperiod:i+1], 1))
                    tr3 = np.abs(low[i-atperiod:i+1] - np.roll(close[i-atperiod:i+1], 1))
                    tr = np.maximum(tr1, np.maximum(tr2, tr3))
                    tr[0] = high[i-atperiod] - low[i-atperiod] if i-atperiod >= 0 else tr1[0]
                    atr_val = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().iloc[-1]
                else:
                    atr_val = 0
                
                if price_close > lowest_since_entry + 3.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0