#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w EMA50 trend filter and volume spike confirmation.
# Long when: Williams %R(14) crosses above -80 (oversold) AND close > 1w EMA50 AND volume > 1.8x 20-bar average
# Short when: Williams %R(14) crosses below -20 (overbought) AND close < 1w EMA50 AND volume > 1.8x 20-bar average
# Exit via ATR(20) trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR
#                      short exit when price > lowest_low_since_entry + 2.0 * ATR
# Uses 1d Williams %R for mean reversion edge (proven in ranging/bear markets), 1w EMA50 for HTF trend alignment, volume spike for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 30-100 total trades over 4 years = 7-25/year.

name = "1d_WilliamsR_1wEMA50_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R (14)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1w EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-bar average volume for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Track position state for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i]):
            continue
            
        # Volume spike condition
        volume_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        # Williams %R crossover signals
        williams_long_signal = williams_r[i-1] <= -80 and williams_r[i] > -80
        williams_short_signal = williams_r[i-1] >= -20 and williams_r[i] < -20
        
        # Trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Update trailing stops
        if position == 1:  # long position
            highest_since_entry = max(highest_since_entry, high[i])
            if close[i] < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            lowest_since_entry = min(lowest_since_entry, low[i])
            if close[i] > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry logic
        if position == 0:  # no position, look for entry
            if williams_long_signal and uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = high[i]
            elif williams_short_signal and downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = low[i]
        elif position == 1 and williams_short_signal:  # reverse long to short on overbought
            signals[i] = -0.25
            position = -1
            lowest_since_entry = low[i]
        elif position == -1 and williams_long_signal:  # reverse short to long on oversold
            signals[i] = 0.25
            position = 1
            highest_since_entry = high[i]
    
    return signals