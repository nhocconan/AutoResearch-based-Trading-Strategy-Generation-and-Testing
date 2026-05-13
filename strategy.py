#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter (EMA34) and volume confirmation.
# Enters long when Bollinger Bands width is at 20-period low (squeeze) and price breaks above upper band with 1d bullish trend (close > EMA34) and volume > 1.5x MA20.
# Enters short when Bollinger Bands width is at 20-period low (squeeze) and price breaks below lower band with 1d bearish trend (close < EMA34) and volume > 1.5x MA20.
# Exits when price returns to the middle Bollinger Band (20-period SMA).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring Bollinger squeeze (low volatility breakout) + HTF trend + volume spike.
# Bollinger squeeze breakouts work in both bull and bear markets as they capture volatility expansion after contraction, often leading to strong trends.

name = "6h_BollingerSqueeze_Breakout_1dTrend_Volume_v1"
timeframe = "6h"
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
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Bollinger Bands (20, 2) on 6h data
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean().values
    dev = close_series.rolling(window=20, min_periods=20).std().values
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    
    # Bollinger Band Width (normalized)
    bb_width = (upper - lower) / basis
    # Squeeze condition: BB width at 20-period low
    bb_width_series = pd.Series(bb_width)
    bb_width_low = bb_width_series.rolling(window=20, min_periods=20).min().values
    squeeze = bb_width <= bb_width_low
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for all indicators
        if np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bollinger squeeze breakout above upper band with 1d bullish trend and volume spike
            if squeeze[i] and close[i] > upper[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bollinger squeeze breakout below lower band with 1d bearish trend and volume spike
            elif squeeze[i] and close[i] < lower[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to middle Bollinger Band (mean reversion)
            if close[i] <= basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to middle Bollinger Band (mean reversion)
            if close[i] >= basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals