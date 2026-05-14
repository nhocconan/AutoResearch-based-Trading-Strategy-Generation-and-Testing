#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses Donchian channel from prior 20 daily bars for structure, weekly EMA50 for trend direction,
# and daily volume spike for conviction. Designed to capture strong breakouts in trending markets
# while avoiding counter-trend trades. Discrete position sizing (0.0, ±0.25) minimizes fee churn.
# Targets 10-25 trades/year per symbol to stay within fee drag limits.

name = "1d_Donchian20_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Indicators (LTF) ---
    # Donchian channel (20) from prior bar
    # Upper = max(high[-20:-1]), Lower = min(low[-20:-1])
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    donchian_high = pd.Series(high_shift).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_shift).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # EMA50 on weekly close
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 1d (wait for completed 1w bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Need 20 bars for Donchian
        # Skip if missing data
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of weekly EMA50
        weekly_trend_up = close[i] > ema_50_1w_aligned[i]
        weekly_trend_down = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND weekly uptrend AND volume spike
            if close[i] > donchian_high[i] and weekly_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND weekly downtrend AND volume spike
            elif close[i] < donchian_low[i] and weekly_trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low (mean reversion) OR loses weekly uptrend
            if close[i] < donchian_low[i] or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high (mean reversion) OR loses weekly downtrend
            if close[i] > donchian_high[i] or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals