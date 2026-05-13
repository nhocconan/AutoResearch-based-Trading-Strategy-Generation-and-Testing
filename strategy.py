#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND volume > 1.5 * avg_volume(20).
# Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND volume > 1.5 * avg_volume(20).
# Exit when price retests the Camarilla pivot (PP) level.
# Uses discrete position sizing (0.30) to balance return and drawdown.
# Designed for ~25-40 trades/year by requiring confluence of breakout, trend, and volume spike.
# Camarilla levels provide precise intraday support/resistance derived from prior day's range.
# Effective in ranging and trending markets by fading extremes with trend alignment.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
timeframe = "4h"
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla: PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1/2
    # S3 = PP - (H - L) * 1.1/2
    pp_1d = (high_1d + low_1d + close_1d) / 3
    r3_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2
    s3_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (use prior completed 1d bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(pp_aligned[i]) or \
           np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(avg_volume[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3, price > 1d EMA34, volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: price breaks below S3, price < 1d EMA34, volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retests pivot point (PP) level
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price retests pivot point (PP) level
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals