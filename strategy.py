#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 1d trend filter and volume confirmation.
# Uses 1d EMA(34) for trend direction, 1h Donchian(20) breakout for entry timing.
# Volume spike filter reduces false signals.
# Long in uptrend when price breaks above Donchian upper band + volume spike.
# Short in downtrend when price breaks below Donchian lower band + volume spike.
# Session filter (08-20 UTC) to avoid low-liquidity hours.
# Target: 15-37 trades/year per symbol (60-150 total) to stay within fee limits.
# Designed to work in both bull and bear markets via trend-following breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # 1h Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: uptrend (close > EMA34) + breakout above upper band + volume spike
            if (close[i] > ema_34_1d_aligned[i] and 
                high[i] > highest_high[i-1] and  # breakout above previous period's high
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: downtrend (close < EMA34) + breakout below lower band + volume spike
            elif (close[i] < ema_34_1d_aligned[i] and 
                  low[i] < lowest_low[i-1] and  # breakout below previous period's low
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: trend reversal or opposite breakout
            if position == 1:
                if (close[i] < ema_34_1d_aligned[i] or low[i] < lowest_low[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if (close[i] > ema_34_1d_aligned[i] or high[i] > highest_high[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_1dEMA34_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0