#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 12h trend filter
# Uses tight entry conditions to limit trades (target: 20-50/year) and avoid fee drag
# Works in bull markets via breakouts and in bear via short breakdowns
# Only trades when volume confirms breakout and higher timeframe trend aligns
name = "4h_DonchianBreakout_VolumeTrend_12h_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for multi-timeframe analysis (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # 4h Donchian channels (20-period)
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ema34_12h_aligned[i]) or \
           np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        if position == 0:
            # Long: breakout above Donchian high + volume + 12h uptrend
            if high[i] > donch_high_20[i-1] and volume_filter and price > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + volume + 12h downtrend
            elif low[i] < donch_low_20[i-1] and volume_filter and price < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below Donchian low or ATR-based stop
            if close[i] < donch_low_20[i] or close[i] < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above Donchian high or ATR-based stop
            if close[i] > donch_high_20[i] or close[i] > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals