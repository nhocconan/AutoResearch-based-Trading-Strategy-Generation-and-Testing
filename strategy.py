#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND close > 1d EMA50 AND volume > 1.5x 20-period average volume.
# Short when price breaks below Donchian(20) low AND close < 1d EMA50 AND volume > 1.5x 20-period average volume.
# Exit on opposite Donchian(10) break or when price crosses 1d EMA50.
# Position size fixed at 0.25 to balance risk and reward, minimizing fee churn.
# Designed for 20-50 trades/year on 4h timeframe by requiring confluence of trend, breakout, and volume.
# Works in bull markets via breakout strength and in bear markets via short breakdowns with trend filter.

name = "4h_DonchianBreakout_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20 for entry, 10 for exit)
    def donchian_channel(high, low, lookback):
        upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
        lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
        return upper, lower
    
    dc_20_upper, dc_20_lower = donchian_channel(high, low, 20)
    dc_10_upper, dc_10_lower = donchian_channel(high, low, 10)
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(dc_20_upper[i]) or np.isnan(dc_20_lower[i]) or \
           np.isnan(dc_10_upper[i]) or np.isnan(dc_10_lower[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian(20) high AND close > 1d EMA50 AND volume confirmation
            if close[i] > dc_20_upper[i] and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian(20) low AND close < 1d EMA50 AND volume confirmation
            elif close[i] < dc_20_lower[i] and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian(10) low OR price crosses below 1d EMA50
            if close[i] < dc_10_lower[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian(10) high OR price crosses above 1d EMA50
            if close[i] > dc_10_upper[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals