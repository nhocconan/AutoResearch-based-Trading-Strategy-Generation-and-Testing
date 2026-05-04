#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trend filter
# Donchian channel breakouts capture institutional participation and momentum bursts.
# Volume spike (2.0x 20-period EMA) confirms genuine breakouts with follow-through.
# ATR(14) trend filter ensures alignment with medium-term momentum to avoid range whipsaws.
# Designed for 4h timeframe targeting 20-50 trades/year (75-200 total over 4 years).
# Discrete sizing (0.25) minimizes fee churn. Works in bull/bear via trend-filtered breakouts.

name = "4h_Donchian20_Breakout_Volume_ATRTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(14) for trend filter and volatility normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no prior close
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Medium-term trend: close vs EMA50 of close
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 for valid EMA50
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(atr[i]) or np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + close > EMA50 (uptrend)
            if (close[i] > donchian_high[i] and volume_spike and 
                close[i] > ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + close < EMA50 (downtrend)
            elif (close[i] < donchian_low[i] and volume_spike and 
                  close[i] < ema_50[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian low OR close < EMA50 (trend change)
            if close[i] < donchian_low[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above Donchian high OR close > EMA50 (trend change)
            if close[i] > donchian_high[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals