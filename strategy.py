#!/usr/bin/env python3
# 4h_donchian_breakout_volume_atr_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and ATR-based trend filter.
# Works in bull markets by capturing upside breakouts and in bear markets by capturing downside breakdowns.
# Volume confirmation reduces false breakouts. ATR trend filter ensures trades align with medium-term momentum.
# Discrete position sizing (0.0, ±0.25) minimizes fee churn. Target: 75-200 trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for trend filter (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 12h HTF data for trend filter (EMA30)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA30 for trend filter
    close_12h = df_12h['close'].values
    ema_30_12h = pd.Series(close_12h).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_30_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ma[i]) or np.isnan(ema_30_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema_30_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema_30_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long breakout: price breaks above Donchian upper with volume
                if close[i] > highest_high[i] and close[i] > ema_30_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price breaks below Donchian lower with volume
                elif close[i] < lowest_low[i] and close[i] < ema_30_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals