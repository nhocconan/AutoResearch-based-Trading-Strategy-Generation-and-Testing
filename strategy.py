#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Targets: 25-40 trades/year by requiring price break above/below Donchian(20) with trend alignment
# Logic: Long when price breaks above Donchian high(20), price > 1d EMA50, volume > 1.5x average
#        Short when price breaks below Donchian low(20), price < 1d EMA50, volume > 1.5x average
#        Exit on opposite Donchian break or trend reversal
# Position size: 0.25 to manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned daily EMA50
        ema_50_i = align_htf_to_ltf(prices, df_1d, ema_50_1d)[i]
        
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_50_i) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: Price breaks above Donchian high, uptrend, volume confirmation
        if position == 0 and close[i] > donch_high[i] and close[i] > ema_50_i and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Price breaks below Donchian low, downtrend, volume confirmation
        elif position == 0 and close[i] < donch_low[i] and close[i] < ema_50_i and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Opposite Donchian break or trend reversal
        elif position != 0:
            if position == 1 and (close[i] < donch_low[i] or close[i] < ema_50_i):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] > donch_high[i] or close[i] > ema_50_i):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0