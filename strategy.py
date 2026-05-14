#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses Donchian channel breakouts for structure, 1d EMA34 for primary trend direction,
# and ATR-normalized volume spike (>1.8x 20-bar average) for conviction.
# Discrete position sizing (0.0, ±0.25) minimizes fee churn. Designed to capture
# strong breakouts in trending markets while avoiding false signals in ranging conditions.
# Targets 20-40 trades/year per symbol (80-160 total over 4 years).

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike_v1"
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
    # ATR(14) for volatility normalization and stoploss reference
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-scaled volume MA: 20-period average of volume / ATR
    vol_atr_ratio = volume / (atr_14 + 1e-10)
    vol_atr_ma_20 = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_atr_ratio > (1.8 * vol_atr_ma_20)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike AND uptrend
            if close[i] > donchian_high[i] and volume_spike[i] and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND volume spike AND downtrend
            elif close[i] < donchian_low[i] and volume_spike[i] and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian low (breakdown) or trend reversal
            if close[i] < donchian_low[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian high (breakout) or trend reversal
            if close[i] > donchian_high[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals