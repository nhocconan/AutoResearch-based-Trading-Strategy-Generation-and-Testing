#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrendRegime_VolumeSpike_v1
Hypothesis: 6h Donchian(20) breakouts with 1d EMA34 trend filter and volume confirmation. 
Donchian provides structure, 1d EMA34 filters regime (bull/bear), volume confirms conviction.
Designed for low trade count (~100 total over 4 years) to minimize fee drag while capturing 
strong breakouts in both bull and bear markets via regime-adaptive logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: highest high over last 20 bars (including current)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over last 20 bars (including current)
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(atr[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime from 1d EMA34
        is_bull = price > ema_34_1d_val
        is_bear = price < ema_34_1d_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long breakouts favored
                long_condition = (price > donch_high_val) and vol_conf
                short_condition = (price < donch_low_val) and vol_conf and (price < ema_34_1d_val * 0.995)  # stricter for shorts
            else:  # bear regime
                # Bear regime: short breakdowns favored
                short_condition = (price < donch_low_val) and vol_conf
                long_condition = (price > donch_high_val) and vol_conf and (price > ema_34_1d_val * 1.005)  # stricter for longs
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 8 bars to reduce churn
            if bars_since_entry < 8:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below Donchian low (failed breakout)
                elif price < donch_low_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks above Donchian high (failed breakdown)
                elif price > donch_high_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dTrendRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0