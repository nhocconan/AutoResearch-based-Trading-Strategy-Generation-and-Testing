#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrendRegime_VolumeSpike_v1
Hypothesis: On 4h timeframe, Donchian(20) breakout combined with 12h EMA50 trend regime and volume confirmation (volume > 1.8x 20-period average) captures strong directional moves. 
In bull regime (12h close > 12h EMA50), favor longs on upper Donchian breakout; in bear regime (12h close < 12h EMA50), favor shorts on lower Donchian breakout. 
Volume confirmation filters weak breakouts. Discrete sizing (0.25) minimizes fee churn. Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 4h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels: highest high and lowest low over 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h volume confirmation (volume > 1.8x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 2 days (8 * 4h = 32h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        regime_ema = ema_50_12h_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_conf = volume_confirmed[i]
        
        # 12h trend regime
        is_bull = price > regime_ema
        is_bear = price < regime_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price breaks above upper Donchian channel
                long_condition = (price > upper_channel) and vol_conf
            else:  # bear regime
                # Bear regime: short when price breaks below lower Donchian channel
                short_condition = (price < lower_channel) and vol_conf
            
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
            
            # Check stoploss (2.5x ATR approximate via Donchian width)
            donchian_width = upper_channel - lower_channel
            if donchian_width > 0:
                approx_atr = donchian_width / 4.0  # rough approximation
                if position == 1:
                    if price < entry_price - 2.5 * approx_atr:
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
                    if price > entry_price + 2.5 * approx_atr:
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
            else:
                # Fallback: time-based exit only
                if bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hTrendRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0