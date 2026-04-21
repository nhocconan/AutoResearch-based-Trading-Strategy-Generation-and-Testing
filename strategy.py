#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrendRegime_VolumeSpike_ATRStop_v1
Hypothesis: 4h Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar MA). 
Trend filter avoids counter-trend whipsaws in bear markets. Volume confirmation ensures breakout validity. 
Discrete sizing (0.25) and ATR-based stoploss (2.0x) reduce churn. Target: 75-200 total trades over 4 years 
by using 4h primary timeframe and tight entry conditions requiring confluence of breakout, trend, and volume. 
Works in bull (breakouts with trend) and bear (faded breakdowns vs trend).
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
    
    # === 1d EMA50 for trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 4h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    # === 4h Donchian channels (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime
        is_bull = price > ema_50_1d_val
        is_bear = price < ema_50_1d_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long breakouts favored
                long_condition = (price > upper_channel) and vol_conf
                short_condition = (price < lower_channel) and vol_conf and (price < ema_50_1d_val * 0.99)  # stricter for shorts
            else:  # bear regime
                # Bear regime: short breakdowns favored
                short_condition = (price < lower_channel) and vol_conf
                long_condition = (price > upper_channel) and vol_conf and (price > ema_50_1d_val * 1.01)  # stricter for longs
            
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
                    signals[i] = 0.25
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
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dTrendRegime_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0