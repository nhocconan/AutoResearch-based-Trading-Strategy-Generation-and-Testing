#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrendRegime_VolumeSpike_v5
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation (>1.8x 20-bar MA). 
In bull 1d regime (close > EMA34), take longs on R1 breakouts; in bear 1d regime (close < EMA34), take shorts on S1 breakouts. 
Volume filter ensures participation. ATR-based stoploss (2.0x) and max hold (8 bars) control risk. 
Discrete sizing (0.25) minimizes fee churn. Target: 20-50 trades/year per symbol.
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
    
    # === 1d EMA34 for daily trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h ATR (12-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=12, min_periods=12).mean().values
    
    # === 4h volume confirmation (volume > 1.8x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_20)
    
    # === 4h Camarilla levels (based on previous day's OHLC) ===
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's OHLC - use 1d data aligned to 4h
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_open = df_1d['open'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    camarilla_R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 4h timeframe (these levels are valid for the entire day)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max hold: 8 * 4h = 32h (~1.3 days)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        r1_val = camarilla_R1_aligned[i]
        s1_val = camarilla_S1_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Daily trend regime
        is_bull = price > ema_34_1d_val
        is_bear = price < ema_34_1d_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long R1 breakout
                long_condition = (price > r1_val) and vol_conf
                short_condition = False  # avoid shorts in bull regime
            else:  # bear regime
                # Bear regime: short S1 breakdown
                short_condition = (price < s1_val) and vol_conf
                long_condition = False  # avoid longs in bear regime
            
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
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
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
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrendRegime_VolumeSpike_v5"
timeframe = "4h"
leverage = 1.0