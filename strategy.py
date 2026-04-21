#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrendRegime_VolumeSpike_v2
Hypothesis: On 4h timeframe, Donchian(20) breakout combined with 1d EMA34 trend regime and volume confirmation (>1.5x 20-period average) captures strong directional moves with low whipsaw. 
In bull regime (1d close > daily EMA34), favor longs on upper Donchian breakout; in bear regime (1d close < daily EMA34), favor shorts on lower Donchian breakout. 
Volume confirmation ensures institutional participation. Discrete sizing (0.30) minimizes fee churn. ATR-based stoploss (2.5x) controls risk. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for daily trend regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for daily trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h volume confirmation (>1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 6  # max 1 day (6 * 4h = 24h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr[i]) if 'atr' in locals() else True or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Calculate ATR for stoploss (using 14-period)
        if i >= 100:  # ATR calculation start
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            atr_val = max(tr1, tr2, tr3)
            # Smooth ATR using Wilder's smoothing (equivalent to RMA)
            if i == 100:
                atr = tr_val
            else:
                atr = (atr_prev * 13 + tr_val) / 14
            atr_prev = atr
        else:
            atr_val = 0.0
            atr_prev = 0.0
            atr = 0.0
        
        price = close[i]
        daily_ema = ema_34_1d_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Daily trend regime
        is_bull = price > daily_ema
        is_bear = price < daily_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long on upper Donchian breakout
                long_condition = (price > donchian_upper[i]) and vol_conf
            else:  # bear regime
                # Bear regime: short on lower Donchian breakout
                short_condition = (price < donchian_lower[i]) and vol_conf
            
            if is_bull and long_condition:
                signals[i] = 0.30
                position = 1
                entry_price = price
                bars_since_entry = 0
                atr_entry = atr  # Store ATR at entry for stoploss
            elif is_bear and short_condition:
                signals[i] = -0.30
                position = -1
                entry_price = price
                bars_since_entry = 0
                atr_entry = atr  # Store ATR at entry for stoploss
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.5x ATR from entry)
            if position == 1:
                if price < entry_price - 2.5 * atr_entry:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if price > entry_price + 2.5 * atr_entry:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dTrendRegime_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0