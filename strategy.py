#!/usr/bin/env python3
# Hypothesis: 4h Donchian channel breakout with 1d trend filter (EMA34), volume confirmation (>1.5x MA20), and ATR-based stoploss.
# Enters long when price breaks above Donchian(20) upper band with bullish 1d trend and volume spike.
# Enters short when price breaks below Donchian(20) lower band with bearish 1d trend and volume spike.
# Exits via ATR trailing stop: long exits when price drops below highest high since entry minus 2.5*ATR(14);
# short exits when price rises above lowest low since entry plus 2.5*ATR(14).
# Uses discrete position sizing (0.30) to balance return and drawdown.
# Designed for moderate trade frequency (~20-50/year) to work in both bull and bear markets by requiring
# strong breakout conditions with volume confirmation and trend alignment, reducing false signals and fee drag.

name = "4h_Donchian_Breakout_1dTrend_Volume_ATRStop"
timeframe = "4h"
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(atr[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper with bullish 1d trend and volume spike
            if close[i] > donchian_upper[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
                highest_high_since_entry = high[i]
            # SHORT: Price breaks below Donchian lower with bearish 1d trend and volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
                lowest_low_since_entry = low[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # EXIT LONG: Price drops below highest high since entry minus 2.5*ATR
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # EXIT SHORT: Price rises above lowest low since entry plus 2.5*ATR
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals