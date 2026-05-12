#!/usr/bin/env python3
"""
1h_4d_1d_HeikinAshi_Trend_Pullback
Hypothesis: Uses 4h Heikin-Ashi candles for trend identification and 1d Heikin-Ashi for higher timeframe confirmation.
Enters long when 4h HA closes above open (bullish candle) with pullback to EMA21 on 1h, and 1d trend is up.
Enters short when 4h HA closes below open (bearish candle) with pullback to EMA21 on 1h, and 1d trend is down.
Uses 1h EMA21 as dynamic support/resistance for pullback entries.
Designed for low trade frequency (~60-150 total trades over 4 years) by requiring multi-timeframe alignment and pullback to moving average.
Works in bull/bear markets by following 1d trend while using 4h HA for entry timing and 1h for precision.
"""

name = "1h_4d_1d_HeikinAshi_Trend_Pullback"
timeframe = "1h"
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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # 1h EMA21 for dynamic support/resistance
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 4h Heikin-Ashi for trend identification
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ha_close_4h = (df_4h['open'] + df_4h['high'] + df_4h['low'] + df_4h['close']) / 4
    ha_open_4h = (df_4h['open'].shift(1, fill_value=df_4h['open'].iloc[0]) + df_4h['close'].shift(1, fill_value=df_4h['close'].iloc[0])) / 2
    ha_high_4h = df_4h[['high', 'open', 'close']].max(axis=1)
    ha_low_4h = df_4h[['low', 'open', 'close']].min(axis=1)
    
    # 4h HA bullish/bearish: close > open = bullish, close < open = bearish
    ha_bullish_4h = ha_close_4h > ha_open_4h
    ha_bearish_4h = ha_close_4h < ha_open_4h
    
    # 1d Heikin-Ashi for higher timeframe trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ha_close_1d = (df_1d['open'] + df_1d['high'] + df_1d['low'] + df_1d['close']) / 4
    ha_open_1d = (df_1d['open'].shift(1, fill_value=df_1d['open'].iloc[0]) + df_1d['close'].shift(1, fill_value=df_1d['close'].iloc[0])) / 2
    
    # 1d HA trend: bullish if close > open, bearish if close < open
    ha_bullish_1d = ha_close_1d > ha_open_1d
    ha_bearish_1d = ha_close_1d < ha_open_1d
    
    # Align all indicators to 1h timeframe
    ema_21_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), ema_21)  # Trivial alignment for same TF
    ha_bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, ha_bullish_4h.astype(float))
    ha_bearish_4h_aligned = align_htf_to_ltf(prices, df_4h, ha_bearish_4h.astype(float))
    ha_bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, ha_bullish_1d.astype(float))
    ha_bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, ha_bearish_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(ema_21_aligned[i]) or
            np.isnan(ha_bullish_4h_aligned[i]) or
            np.isnan(ha_bearish_4h_aligned[i]) or
            np.isnan(ha_bullish_1d_aligned[i]) or
            np.isnan(ha_bearish_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 4h HA bullish, 1d HA bullish, price pulls back to EMA21 from below
            if (ha_bullish_4h_aligned[i] and 
                ha_bullish_1d_aligned[i] and
                close[i] >= ema_21_aligned[i] * 0.998 and  # Allow small deviation below EMA
                close[i] <= ema_21_aligned[i] * 1.002):    # Allow small deviation above EMA
                signals[i] = 0.20
                position = 1
            # SHORT: 4h HA bearish, 1d HA bearish, price pulls back to EMA21 from above
            elif (ha_bearish_4h_aligned[i] and 
                  ha_bearish_1d_aligned[i] and
                  close[i] >= ema_21_aligned[i] * 0.998 and  # Allow small deviation below EMA
                  close[i] <= ema_21_aligned[i] * 1.002):    # Allow small deviation above EMA
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h HA turns bearish OR 1d HA turns bearish
            if (not ha_bullish_4h_aligned[i]) or (not ha_bullish_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h HA turns bullish OR 1d HA turns bullish
            if (not ha_bearish_4h_aligned[i]) or (not ha_bearish_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals