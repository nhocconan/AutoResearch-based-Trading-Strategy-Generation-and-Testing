#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volume spike and EMA trend filter.
# Uses Donchian channel (20) from prior 4h for structure, ATR-normalized volume spike (>1.5x 20-bar ATR-scaled avg volume) on 1d for conviction,
# and EMA34 > EMA89 on 1d to ensure bullish/bearish alignment. Discrete position sizing (0.0, ±0.25) minimizes fee churn.
# Designed to capture strong breakouts in trending markets while avoiding false signals in ranging conditions. Targets 20-50 trades/year per symbol.

name = "4h_Donchian20_Breakout_1dATRVolumeSpike_EMATrend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # Donchian channel (20) from prior bar
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # ATR(14) for volatility normalization on 1d
    high_shift_1d = np.roll(high_1d, 1)
    low_shift_1d = np.roll(low_1d, 1)
    close_shift_1d = np.roll(close_1d, 1)
    high_shift_1d[0] = high_1d[0]
    low_shift_1d[0] = low_1d[0]
    close_shift_1d[0] = close_1d[0]
    
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - close_shift_1d), np.abs(low_1d - close_shift_1d)))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-scaled volume MA: 20-period average of volume / ATR on 1d
    vol_atr_ratio_1d = volume_1d / (atr_14_1d + 1e-10)
    vol_atr_ma_20_1d = pd.Series(vol_atr_ratio_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_atr_ratio_1d > (1.5 * vol_atr_ma_20_1d)
    
    # EMA trend filter on 1d: EMA34 > EMA89 for bullish, < for bearish
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align HTF indicators to 4h (wait for completed 1d bar)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_89_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when EMA34 and EMA89 are aligned
        is_bullish = ema_34_1d_aligned[i] > ema_89_1d_aligned[i]
        is_bearish = ema_34_1d_aligned[i] < ema_89_1d_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian upper AND volume spike AND bullish trend
            if close[i] > highest_20[i] and volume_spike_1d_aligned[i] and is_bullish:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower AND volume spike AND bearish trend
            elif close[i] < lowest_20[i] and volume_spike_1d_aligned[i] and is_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian lower (stoploss) OR opposite Donchian break
            if close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian upper (stoploss) OR opposite Donchian break
            if close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals