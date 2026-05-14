#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d volume spike (ATR-normalized) and 1d EMA34 trend filter.
# Uses Donchian channel breakouts for structure, ATR-normalized volume spike (>1.5x 20-bar average) for conviction,
# and 1d EMA34 for trend direction (long only when price > EMA34, short only when price < EMA34).
# Discrete position sizing (0.0, ±0.30) minimizes fee churn. Targets 20-50 trades/year per symbol.

name = "4h_Donchian20_Breakout_1dEMA34_ATRVolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility normalization
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-normalized volume and its 20-bar MA
    vol_atr_ratio = volume / (atr_14 + 1e-10)
    vol_atr_ma_20 = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_atr_ratio > (1.5 * vol_atr_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # EMA(34) on 1d close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike AND price > 1d EMA34 (uptrend)
            if close[i] > highest_20[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian low AND volume spike AND price < 1d EMA34 (downtrend)
            elif close[i] < lowest_20[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian low (mean reversion) OR trend change
            if close[i] < lowest_20[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian high (mean reversion) OR trend change
            if close[i] > highest_20[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals