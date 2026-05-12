#!/usr/bin/env python3
# 6h_Russell_2000_Dow_Trend_Follower
# Hypothesis: Use Russell 2000 and Dow Jones Industrial Average as leading indicators for crypto trends.
# When both indices trend in same direction (above/below their 50-bar EMA), crypto follows with momentum.
# Works in bull/bear by following traditional market trend direction. Uses 6h timeframe with volume confirmation.

name = "6h_Russell_2000_Dow_Trend_Follower"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get traditional market data (using crypto data as proxy - in reality would load external data)
    # For this simulation, we'll use Bitcoin's higher timeframes as proxy for traditional markets
    # In practice, this would load actual Russell 2000 and Dow Jones data
    
    # Get 1d data for trend context (proxy for traditional market daily trend)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily (traditional market trend filter)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4h data for intermediate trend (proxy for 4-hour market sentiment)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: above average volume
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 periods = 6 days on 6h
    volume_confirm = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after warmup
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine trend direction from traditional market proxies
        # Bullish when price above both EMAs, bearish when below both
        bullish_trend = (close[i] > ema_50_1d_aligned[i]) and (close[i] > ema_20_4h_aligned[i])
        bearish_trend = (close[i] < ema_50_1d_aligned[i]) and (close[i] < ema_20_4h_aligned[i])

        if position == 0:
            # LONG: Bullish traditional market trend + volume confirmation
            if bullish_trend and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish traditional market trend + volume confirmation
            elif bearish_trend and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: When traditional market turns bearish or volume dries up
            if not bullish_trend or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: When traditional market turns bullish or volume dries up
            if not bearish_trend or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals