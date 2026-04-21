#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrendRegime_VolumeSpike_v1
Hypothesis: On 12h timeframe, Camarilla pivot R1/S1 breakouts combined with 1d EMA34 trend regime and volume confirmation (volume > 1.8x 20-period average) capture institutional moves in both bull and bear markets. 
In bull regime (1d close > 1d EMA34), buy R1 breakouts with volume. In bear regime (1d close < 1d EMA34), sell S1 breakdowns with volume. 
Discrete sizing (0.25) and ATR-based stoploss (2.5x ATR) control drawdown. Target: 50-150 total trades over 4 years.
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
    
    # === 12h Camarilla pivot levels (R1, S1) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate pivot from previous bar
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First bar: use same values (no look-ahead)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + 1.1 * (prev_high - prev_low) / 12.0
    s1 = pivot - 1.1 * (prev_high - prev_low) / 12.0
    
    # === 12h volume confirmation (volume > 1.8x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_20)
    
    # === 12h ATR for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 4 days (8 * 12h = 96h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_confirmed[i]) or np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        regime_ema = ema_34_1d_aligned[i]
        r1_level = r1[i]
        s1_level = s1[i]
        vol_conf = volume_confirmed[i]
        atr_val = atr[i]
        
        # Weekly trend regime
        is_bull = price > regime_ema
        is_bear = price < regime_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price breaks above R1 with volume
                long_condition = (price > r1_level) and vol_conf
            else:  # bear regime
                # Bear regime: short when price breaks below S1 with volume
                short_condition = (price < s1_level) and vol_conf
            
            if is_bull and long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif is_bear and short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr_val:
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
                if price > entry_price + 2.5 * atr_val:
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrendRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0