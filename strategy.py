#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrendRegime_VolumeSpike_v1
Hypothesis: 12h Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation (>1.8x 20-bar MA). 
In bull regime (price > 1d EMA50), take longs on upper band breakouts; in bear regime (price < 1d EMA50), take shorts on lower band breakdowns. 
Volume confirmation reduces false breakouts. ATR-based stoploss (2.5x) and discrete sizing (0.25) control risk. 
Target: 50-150 total trades over 4 years by requiring confluence of breakout, trend, and volume. 
Designed to work in bull (breakouts with trend) and bear (faded breakdowns vs trend) markets with volume filter to control overtrading.
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
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d EMA50 for trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 12h volume confirmation (volume > 1.8x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_20)
    
    # === 12h Donchian channels (20-period) based on PREVIOUS bar's high/low ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = prev_low[0] = np.nan  # first bar invalid
    
    upper_channel = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime
        is_bull = price > ema_50_1d_val
        is_bear = price < ema_50_1d_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long breakouts favored
                long_condition = (price > upper_val) and vol_conf
                short_condition = (price < lower_val) and vol_conf and (price < ema_50_1d_val * 0.99)  # stricter for shorts
            else:  # bear regime
                # Bear regime: short breakdowns favored
                short_condition = (price < lower_val) and vol_conf
                long_condition = (price > upper_val) and vol_conf and (price > ema_50_1d_val * 1.01)  # stricter for longs
            
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
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below lower channel (failed breakout)
                elif price < lower_val:
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
                # Exit if price breaks above upper channel (failed breakdown)
                elif price > upper_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dTrendRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0