#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloudFilter_VolumeSpike
Hypothesis: On 6h timeframe, Ichimoku Tenkan-Kijun (TK) cross aligned with 1d cloud (bullish: price above cloud, bearish: price below cloud) and volume spikes (>2.0x 20-period MA) capture high-probability trend continuation moves. The cloud acts as a dynamic support/resistance filter reducing false signals. Uses discrete position sizing (0.0, ±0.25) and 6h ATR-based trailing stop (2.5x) for exits. Targets 12-25 trades/year by requiring HTF cloud alignment, volume confirmation, and Ichimoku momentum—designed to work in both bull (trend continuation) and bear (trend continuation down) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d Ichimoku cloud: Senkou Span A and B
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2).shift(26)
    
    # Align cloud components to 6h timeframe (wait for completed 1d bar)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    # 6h Ichimoku: Tenkan-Kijun cross for entry signal
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_6h = (high_9_6h + low_9_6h) / 2
    
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_6h = (high_26_6h + low_26_6h) / 2
    
    # TK cross: bullish when Tenkan > Kijun, bearish when Tenkan < Kijun
    tk_bullish = tenkan_6h > kijun_6h
    tk_bearish = tenkan_6h < kijun_6h
    
    # 6h ATR(14) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = tr_6h.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_6h_values = atr_6h.values
    
    # Volume spike filter: volume > 2.0 * 20-period MA on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of Ichimoku (52), ATR (14), volume MA (20)
    start_idx = max(52, 14, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        tk_bull = tk_bullish[i]
        tk_bear = tk_bearish[i]
        vol_spike = volume_spike[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        atr_val = atr_6h_values[i]
        
        # Skip if any data not ready
        if (np.isnan(senkou_a_val) or np.isnan(senkou_b_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Cloud filter: price above cloud = bullish bias, price below cloud = bearish bias
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        # Entry conditions: TK cross in direction of cloud + volume spike
        long_entry = tk_bull and price_above_cloud and vol_spike
        short_entry = tk_bear and price_below_cloud and vol_spike
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.5 * ATR
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.5 * ATR
            stop_price = lowest_since_short + 2.5 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloudFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0