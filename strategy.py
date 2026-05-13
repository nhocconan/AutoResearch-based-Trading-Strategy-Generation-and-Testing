#!/usr/bin/env python3
# 4h_Combined_Trend_Momentum_With_Volume_Confirmation
# Hypothesis: Combines multiple trend and momentum indicators to filter noise and capture sustained moves.
# Uses 4h EMA crossover for trend direction, RSI for momentum strength, and volume spike for confirmation.
# Works in bull/bear markets by requiring alignment across timeframes and momentum filters.
# Designed to limit trades to 20-50 per year to minimize fee drag.

name = "4h_Combined_Trend_Momentum_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h EMA crossover: EMA12 > EMA26 for bullish, EMA12 < EMA26 for bearish
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema12[i]) or 
            np.isnan(ema26[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above 1D EMA50 + bullish 4H EMA crossover + RSI > 50 + volume spike
            if (close[i] > ema50_1d_aligned[i] and 
                ema12[i] > ema26[i] and 
                rsi[i] > 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below 1D EMA50 + bearish 4H EMA crossover + RSI < 50 + volume spike
            elif (close[i] < ema50_1d_aligned[i] and 
                  ema12[i] < ema26[i] and 
                  rsi[i] < 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below 1D EMA50 or bearish EMA crossover or RSI < 40
            if (close[i] < ema50_1d_aligned[i] or 
                ema12[i] < ema26[i] or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above 1D EMA50 or bullish EMA crossover or RSI > 60
            if (close[i] > ema50_1d_aligned[i] or 
                ema12[i] > ema26[i] or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals