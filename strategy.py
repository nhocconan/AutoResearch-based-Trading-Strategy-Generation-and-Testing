#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50), volume confirmation (1.8x MA20), and ATR-based trailing stop.
# Enters long when price breaks above Donchian upper band with 1d bullish trend (close > EMA50), volume > 1.8x MA20.
# Enters short when price breaks below Donchian lower band with 1d bearish trend (close < EMA50), volume > 1.8x MA20.
# Exits when price reverses to Donchian midpoint or ATR trailing stop (3 * ATR(14) from extreme).
# Uses discrete position sizing (0.30) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~20-50/year) by requiring strict confluence: breakout + HTF trend + volume spike.
# Donchian channels provide objective structure, while HTF trend filter avoids counter-trend trades.
# Higher volume threshold (1.8x) reduces false breakouts, improving signal quality in both bull and bear markets.

name = "4h_Donchian20_Breakout_1dTrend_Volume_v1"
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period) from 1d data
    # Donchian upper = max(high, 20), lower = min(low, 20) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Volume filter: current volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
    # ATR(14) for trailing stop
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track entry price and extreme for trailing stop
    entry_price = np.full(n, np.nan)
    extreme_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or \
           np.isnan(donchian_mid_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or \
           np.isnan(vol_ma20[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper with 1d bullish trend and volume spike
            if close[i] > donchian_upper_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
                entry_price[i] = close[i]
                extreme_price[i] = close[i]
            # SHORT: Price breaks below Donchian lower with 1d bearish trend and volume spike
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
                entry_price[i] = close[i]
                extreme_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Update extreme for long position
            extreme_price[i] = max(extreme_price[i-1], close[i])
            # EXIT LONG: Price reverses to Donchian midpoint OR ATR trailing stop hit
            if close[i] < donchian_mid_aligned[i] or close[i] < extreme_price[i] - 3.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
                extreme_price[i] = np.nan
            else:
                signals[i] = 0.30
                entry_price[i] = entry_price[i-1]
        elif position == -1:
            # Update extreme for short position
            extreme_price[i] = min(extreme_price[i-1], close[i])
            # EXIT SHORT: Price reverses to Donchian midpoint OR ATR trailing stop hit
            if close[i] > donchian_mid_aligned[i] or close[i] > extreme_price[i] + 3.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
                extreme_price[i] = np.nan
            else:
                signals[i] = -0.30
                entry_price[i] = entry_price[i-1]
    
    return signals