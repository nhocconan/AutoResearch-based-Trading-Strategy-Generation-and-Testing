#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based volume spike.
# Uses Donchian channel (20-period high/low) from prior 1d for structure, ATR-normalized volume spike (>1.8x 20-bar ATR-scaled avg volume) for conviction,
# and 1w EMA50 > EMA200 to ensure bullish long-term trend (avoid shorts in bear markets). Discrete position sizing (0.0, ±0.25) minimizes fee churn.
# Designed to capture strong breakouts in bull markets while avoiding bearish conditions. Targets 15-30 trades/year per symbol.

name = "1d_Donchian20_Breakout_1wEMA50Trend_ATRVolumeSpike_v1"
timeframe = "1d"
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
    
    # --- 1d Indicators (LTF) ---
    # ATR(14) for volatility normalization
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
    
    # Donchian channel (20) from prior 1d bar
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = np.roll(high_roll, 1)  # prior bar's 20-period high
    donchian_low = np.roll(low_roll, 1)    # prior bar's 20-period low
    donchian_high[0] = high[0]
    donchian_low[0] = low[0]
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # EMA50 and EMA200 on 1w
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to 1d (wait for completed 1w bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Bullish trend filter: EMA50 > EMA200
    bullish_trend = ema_50_aligned > ema_200_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(atr_14[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(bullish_trend[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in bullish long-term trend (avoid shorts in bear markets)
        if not bullish_trend[i]:
            # In bearish/neutral trend, stay flat or exit longs
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                # Exit long if price touches Donchian low (mean reversion)
                if close[i] <= donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # No shorts allowed in bearish trend - force flat
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish trend: look for long breakouts with volume confirmation
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike
            if close[i] > donchian_high[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian low (mean reversion)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals