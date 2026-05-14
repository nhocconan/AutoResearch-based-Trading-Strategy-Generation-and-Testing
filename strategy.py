#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ADX trend filter.
# Uses 12h Donchian channels (20-period) for breakout structure, 1d volume spike (>1.5x 20-bar EMA) for conviction,
# and ADX > 25 on 12h to ensure trending markets. Discrete position sizing (0.0, ±0.25) minimizes fee churn.
# Designed to capture strong breakouts in trending markets while avoiding false signals in ranging conditions.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue). Targets 12-25 trades/year per symbol.

name = "12h_Donchian20_Breakout_1dVolumeSpike_ADXFilter_v1"
timeframe = "12h"
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
    
    # --- 12h Indicators (LTF) ---
    # ATR(14) for volatility (used in ADX)
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ADX (14) for trend strength on 12h
    plus_dm = np.where((high - high_shift) > (low_shift - low), np.maximum(high - high_shift, 0), 0)
    minus_dm = np.where((low_shift - low) > (high - high_shift), np.maximum(low_shift - low, 0), 0)
    
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Volume spike: current 1d volume > 1.5x 20-bar EMA of volume
    volume_ema_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * volume_ema_20)
    
    # Align volume spike to 12h (wait for completed 1d bar)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Donchian channels (20-period) on 12h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(adx[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        if adx[i] <= 25:
            # In ranging/weak trend, stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                # Exit long if price crosses below midpoint (mean reversion)
                midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
                if close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if price crosses above midpoint (mean reversion)
                midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
                if close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            continue
        
        # Trending regime: look for breakouts with volume confirmation
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike
            if close[i] > highest_high_20[i] and volume_spike_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND volume spike
            elif close[i] < lowest_low_20[i] and volume_spike_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian low (stop and reverse)
            if close[i] < lowest_low_20[i]:
                signals[i] = -0.25
                position = -1
            # EXIT LONG: Price crosses below midpoint (mean reversion exit)
            elif close[i] < (highest_high_20[i] + lowest_low_20[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian high (stop and reverse)
            if close[i] > highest_high_20[i]:
                signals[i] = 0.25
                position = 1
            # EXIT SHORT: Price crosses above midpoint (mean reversion exit)
            elif close[i] > (highest_high_20[i] + lowest_low_20[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals