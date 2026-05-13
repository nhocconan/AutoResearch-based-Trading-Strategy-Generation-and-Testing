#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Donchian(20) breakout direction and 1d trend filter (EMA50) with volume confirmation (1.5x MA20).
# Enters long when price breaks above 4h Donchian high, 1d trend is bullish (close > EMA50), and volume > 1.5x MA20.
# Enters short when price breaks below 4h Donchian low, 1d trend is bearish (close < EMA50), and volume > 1.5x MA20.
# Exits when price crosses the 4h EMA20 (mean reversion).
# Uses discrete position sizing (0.20) to limit fee churn and manage drawdown.
# Target: 15-37 trades/year by requiring strict confluence of 4h breakout, 1d trend, and volume spike.
# Works in bull markets via breakouts and in bear markets via short entries aligned with 1d trend.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.

name = "1h_Donchian20_Breakout_1dTrend_Volume_v1"
timeframe = "1h"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels (based on previous 4h bar)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels: upper, lower (based on previous 4h bar)
    lookback = 20
    donchian_high = pd.Series(high_4h).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low_4h).rolling(window=lookback, min_periods=lookback).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # 4h EMA20 for exit condition
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(ema20_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with 1d bullish trend and volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Donchian low with 1d bearish trend and volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 4h EMA20 (mean reversion)
            if close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above 4h EMA20 (mean reversion)
            if close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals