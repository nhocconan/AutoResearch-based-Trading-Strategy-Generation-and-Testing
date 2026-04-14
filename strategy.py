#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Donchian(20) breakout identifies breakouts in both directions
# 1d EMA50 filter ensures we trade with the higher timeframe trend
# Volume confirmation requires volume > 1.5x 20-period EMA to filter false breakouts
# ATR-based stoploss (exit when price moves against position by 2*ATR)
# Designed for 20-40 trades/year with controlled risk in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Calculate ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA (50-period) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned 1d EMA
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)[i]
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_1d_aligned) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:  # No position - look for breakout entries
            # Long breakout: price breaks above Donchian high with volume and uptrend
            if close[i] > donchian_high[i] and volume_confirm and close[i] > ema_1d_aligned:
                position = 1
                signals[i] = position_size
            # Short breakout: price breaks below Donchian low with volume and downtrend
            elif close[i] < donchian_low[i] and volume_confirm and close[i] < ema_1d_aligned:
                position = -1
                signals[i] = -position_size
        elif position == 1:  # Long position - exit on reversal or stoploss
            # Exit if price breaks below Donchian low (reversal signal)
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            # Stoploss: exit if price moves against position by 2*ATR
            elif close[i] < close[i-1] - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit on reversal or stoploss
            # Exit if price breaks above Donchian high (reversal signal)
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            # Stoploss: exit if price moves against position by 2*ATR
            elif close[i] > close[i-1] + 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_1dTrend_Volume_ATR"
timeframe = "4h"
leverage = 1.0