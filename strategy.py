#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_ATRStop_v1
Hypothesis: 4h Donchian(20) breakouts aligned with 12h trend (EMA50) and volume confirmation (2.0x 20-bar MA) with ATR(14) stoploss (2.0x). Designed for low trade count (<150/4 years) and robustness in bull/bear regimes via trend filter and volatility-based exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for trend regime)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h EMA50 for trend regime ===
    close_12h = df_12h['close'].values
    ema_50_12h = calculate_ema(close_12h, 50)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 4h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    # === 4h Donchian channels (20-period) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime
        is_bull = price > ema_50_12h_val
        is_bear = price < ema_50_12h_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long breakouts favored
                long_condition = (price > upper_channel) and vol_conf
                short_condition = (price < lower_channel) and vol_conf and (price < ema_50_12h_val * 0.995)  # stricter for shorts
            else:  # bear regime
                # Bear regime: short breakdowns favored
                short_condition = (price < lower_channel) and vol_conf
                long_condition = (price > upper_channel) and vol_conf and (price > ema_50_12h_val * 1.005)  # stricter for longs
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.30
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = 0.30 if position == 1 else -0.30
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below lower channel (failed breakout)
                elif price < lower_channel:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks above upper channel (failed breakdown)
                elif price > upper_channel:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0