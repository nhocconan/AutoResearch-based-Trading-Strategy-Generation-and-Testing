#!/usr/bin/env python3
# 4h_donchian_breakout_volume_atr_v1
# Hypothesis: 4h Donchian breakout with volume confirmation and ATR-based stoploss.
# Long: Price breaks above 20-period Donchian high, volume > 1.5x 20-period average.
# Short: Price breaks below 20-period Donchian low, volume > 1.5x 20-period average.
# Exit: ATR trailing stop (close < highest_high_since_entry - 3*ATR for longs, 
#       close > lowest_low_since_entry + 3*ATR for shorts).
# Uses 1d trend filter (price > 50 EMA for longs, price < 50 EMA for shorts) to 
# avoid counter-trend trades. Target: 19-50 trades/year (75-200 total) on BTC/ETH/SOL.
# Works in bull (trend-following breaks) and bear (mean-reversion at extremes with volume).

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
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_high = 0.0  # highest high since entry for longs
    entry_low = 0.0   # lowest low since entry for shorts
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > entry_high:
                entry_high = high[i]
            # ATR trailing stop: close < entry_high - 3*ATR
            if close[i] < entry_high - 3.0 * atr[i]:
                position = 0
                entry_high = 0.0
                signals[i] = 0.0
            # Exit: reverse signal (price breaks below Donchian low)
            elif low[i] < donchian_low[i]:
                position = 0
                entry_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < entry_low:
                entry_low = low[i]
            # ATR trailing stop: close > entry_low + 3*ATR
            if close[i] > entry_low + 3.0 * atr[i]:
                position = 0
                entry_low = 0.0
                signals[i] = 0.0
            # Exit: reverse signal (price breaks above Donchian high)
            elif high[i] > donchian_high[i]:
                position = 0
                entry_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high, volume confirmed, price > 1d EMA50
            if (high[i] > donchian_high[i] and volume_confirmed and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                entry_high = high[i]
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low, volume confirmed, price < 1d EMA50
            elif (low[i] < donchian_low[i] and volume_confirmed and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                entry_low = low[i]
                signals[i] = -0.25
    
    return signals