#!/usr/bin/env python3
"""
4h_KAMA_Direction_1dTrendRegime_VolumeSpike_v1
Hypothesis: 4h KAMA direction (trend) aligned with 1d EMA34 trend regime and volume confirmation (>2x 20-bar MA). 
In bull regime (price > 1d EMA34), take longs when KAMA turns up; in bear regime (price < 1d EMA34), take shorts when KAMA turns down. 
ATR-based stoploss (2.0x) and discrete sizing (0.25) reduce churn. Target: 75-200 total trades over 4 years by requiring confluence of KAMA turn, trend regime, and volume. 
Designed to work in bull (trend following with KAMA) and bear (counter-trend KAMA turns vs higher timeframe) markets.
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
    
    # === 4h KAMA (ER=10, fast=2, slow=30) for trend direction ===
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(10).values)  # 10-period net change
    vol = np.abs(close_s.diff(1).values)      # 1-period volatility
    sum_vol = pd.Series(vol).rolling(window=10, min_periods=10).sum().values
    er = np.where(sum_vol > 0, change / sum_vol, 0)  # efficiency ratio
    sc = (er * (2.0/2 - 2.0/30) + 2.0/30) ** 2     # smoothing constant
    kama = np.full_like(close, np.nan)
    kama[9] = close_s.iloc[9]  # seed value at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA turning points: up when current > previous, down when current < previous
    kama_up = kama > np.roll(kama, 1)
    kama_down = kama < np.roll(kama, 1)
    kama_up[0] = False
    kama_down[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(kama[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        vol_conf = volume_confirmed[i]
        kama_up_val = kama_up[i]
        kama_down_val = kama_down[i]
        
        # Trend regime
        is_bull = price > ema_34_1d_val
        is_bear = price < ema_34_1d_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long when KAMA turns up
                long_condition = kama_up_val and vol_conf
                short_condition = False  # avoid shorts in bull regime
            else:  # bear regime
                # Bear regime: short when KAMA turns down
                short_condition = kama_down_val and vol_conf
                long_condition = False   # avoid longs in bear regime
            
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
                # Exit if KAMA turns down (trend change)
                elif kama_down[i]:
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
                # Exit if KAMA turns up (trend change)
                elif kama_up[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_1dTrendRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0