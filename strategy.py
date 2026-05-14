#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Uses 20-bar Donchian channels from prior 1d for structure, volume > 1.5x 20-bar average for conviction,
# and 1w EMA34 > EMA50 to ensure bullish higher-timeframe trend. Discrete position sizing (0.0, ±0.30) minimizes fee churn.
# Designed to capture strong breakouts in bullish higher-timeframe trends while avoiding signals in bearish regimes.
# Targets 15-25 trades/year per symbol.

name = "1d_Donchian20_Breakout_1wEMATrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Indicators (LTF) ---
    # Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # EMA34 and EMA50 on 1w
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 1d (wait for completed 1w bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when 1w EMA34 > EMA50 (bullish higher-timeframe trend)
        if ema_34_1w_aligned[i] <= ema_50_1w_aligned[i]:
            # In bearish/neutral 1w trend, stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                # Exit long if price breaks below Donchian low (stoploss)
                if close[i] < lowest_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            elif position == -1:
                # Should not happen in bullish filter, but exit if price breaks above Donchian high
                if close[i] > highest_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
            continue
        
        # Bullish 1w regime: look for long breakouts with volume confirmation
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike
            if close[i] > highest_20[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low (ATR-based stoploss equivalent)
            if close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
    
    return signals