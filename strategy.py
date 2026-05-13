#!/usr/bin/env python3
# Hypothesis: 12h Candlestick Pattern + 1-week RSI Trend Filter
# Uses bullish/bearish engulfing patterns on 12h chart for entry signals.
# Confirms with 1-week RSI: only long when weekly RSI > 50, short when weekly RSI < 50.
# Adds volume confirmation: current volume > 20-period average.
# Exits when opposite engulfing pattern forms or RSI crosses 50 level.
# Designed for low trade frequency (<30/year) to minimize fee drift.
# Works in both bull and bear markets by following higher timeframe momentum.

name = "12h_Engulfing_WeeklyRSI_Filter"
timeframe = "12h"
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
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (close > open_) & (open_ > close) & (close >= open_.shift(1)) & (open_ <= close.shift(1))
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulf = (open_ > close) & (close < open_) & (open_ >= close.shift(1)) & (close <= open_.shift(1))
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    # Get 1-week RSI for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = pd.Series(df_1w['close'].values)
    delta = close_1w.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_1w = (100 - (100 / (1 + rs))).values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for volume MA
        if np.isnan(vol_ma20[i]) or np.isnan(rsi_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish engulfing + weekly RSI > 50 + volume confirmation
            if bullish_engulf.iloc[i] and rsi_1w_aligned[i] > 50 and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish engulfing + weekly RSI < 50 + volume confirmation
            elif bearish_engulf.iloc[i] and rsi_1w_aligned[i] < 50 and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish engulfing forms OR weekly RSI < 50
            if bearish_engulf.iloc[i] or rsi_1w_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish engulfing forms OR weekly RSI > 50
            if bullish_engulf.iloc[i] or rsi_1w_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals