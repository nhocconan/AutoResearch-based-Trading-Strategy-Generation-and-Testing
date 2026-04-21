#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_HTFTrend_VolumeRegime_ATRStop
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume regime (choppiness < 50).
Enter long when price breaks above 20-period high with 1d uptrend and low chop (trending regime).
Enter short when price breaks below 20-period low with 1d downtrend and low chop.
Exit on ATR(14) trailing stop (2.5*ATR) or opposite Donchian break.
Designed for moderate trade frequency (~30-50/year) to balance edge and fee drag.
Works in bull/bear via 1d trend alignment and chop regime filter to avoid whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend, 1w for regime/context)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d EMA34 for HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d Chopiness Index (14) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d_arr, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d_arr, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Sum of true range over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Chopiness Index: 100 * log10(sum_tr_14 / range_14) / log10(14)
    # Avoid division by zero and log of zero
    chop_1d = np.where(
        (range_14 > 0) & (sum_tr_14 > 0),
        100 * np.log10(sum_tr_14 / range_14) / np.log10(14),
        50.0  # default to neutral when invalid
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) 
            or np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Regime filter: chop < 50 indicates trending market (not ranging)
            trending_regime = chop_1d_aligned[i] < 50.0
            
            # Volume confirmation: current volume > 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_confirm = volume[i] > vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > Donchian high, 1d uptrend, trending regime, volume spike
            long_breakout = price > donch_high[i]
            long_trend = price > ema_34_1d_aligned[i]
            
            # Short conditions: price < Donchian low, 1d downtrend, trending regime, volume spike
            short_breakout = price < donch_low[i]
            short_trend = price < ema_34_1d_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and trending_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and trending_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price closes below Donchian low (breakdown)
            elif price < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price closes above Donchian high (breakout)
            elif price > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_HTFTrend_VolumeRegime_ATRStop"
timeframe = "4h"
leverage = 1.0