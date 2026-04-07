#!/usr/bin/env python3
"""
6h_keltner_channel_1w_trend_volume_v1
Hypothesis: Keltner Channel breakouts with weekly trend alignment and volume confirmation capture
institutional moves while avoiding false breakouts in ranging markets. Weekly trend filter ensures
trading with the higher-timeframe momentum, reducing whipsaws. Targets 12-37 trades/year by
requiring confluence of upper/lower band break, volume spike (>1.5x 20-period average), and
weekly EMA trend alignment. Works in bull markets (breakouts continue) and bear markets
(breakdowns continue) by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_channel_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly OHLC for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Keltner Channel (20, 2.0) on 6h timeframe
    # Middle line = EMA20
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Average True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Upper and lower bands
    upper_band = ema20 + 2.0 * atr
    lower_band = ema20 - 2.0 * atr
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(ema20[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below middle line (EMA20) OR trend turns down
            if close[i] < ema20[i] or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above middle line (EMA20) OR trend turns up
            if close[i] > ema20[i] or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above upper band + volume + weekly uptrend
            if (close[i] > upper_band[i] and 
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower band + volume + weekly downtrend
            elif (close[i] < lower_band[i] and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals