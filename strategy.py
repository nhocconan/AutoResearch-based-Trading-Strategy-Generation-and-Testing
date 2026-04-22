#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation.
# Uses 1d EMA(34) for trend direction, 4h Donchian(20) for breakout signals.
# Volume spike filter reduces false signals.
# Long in uptrend when price breaks above Donchian upper band + volume spike.
# Short in downtrend when price breaks below Donchian lower band + volume spike.
# Session filter (08-20 UTC) to avoid low-liquidity hours.
# Target: 20-50 trades/year per symbol (80-200 total) to stay within fee limits.
# Designed to work in both bull and bear markets via trend-following breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Donchian(20)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: uptrend (price > EMA34) + Donchian breakout up + volume spike
            if (close[i] > ema_34_1d_aligned[i] and 
                close[i] > donchian_high_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend (price < EMA34) + Donchian breakout down + volume spike
            elif (close[i] < ema_34_1d_aligned[i] and 
                  close[i] < donchian_low_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend reversal or Donchian reversal
            if position == 1:
                if (close[i] < ema_34_1d_aligned[i] or close[i] < donchian_low_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > ema_34_1d_aligned[i] or close[i] > donchian_high_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_Session"
timeframe = "4h"
leverage = 1.0