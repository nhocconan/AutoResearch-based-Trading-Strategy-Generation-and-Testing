#!/usr/bin/env python3
"""
12h_price_channel_breakout_v2
Hypothesis: On 12h timeframe, enter long when price breaks above 20-period Donchian upper band with above-average volume and 1d ADX > 25 (trending market), enter short when price breaks below 20-period Donchian lower band with above-average volume and 1d ADX > 25. Exit when price crosses back below the 10-period EMA (for longs) or above the 10-period EMA (for shorts). Uses 1d ADX trend filter to avoid choppy markets. Designed for 15-30 trades/year to minimize fee decay while capturing trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_price_channel_breakout_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    if len(high) < 20:
        return np.zeros(n)
    
    # Donchian upper and lower bands
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing = alpha = 1/period)
    def wilde_rma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])  # Skip first NaN
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilde_rma(tr, 14)
    plus_di = 100 * wilde_rma(plus_dm, 14) / atr
    minus_di = 100 * wilde_rma(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilde_rma(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_10[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trend_ok = adx_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: price crosses below 10-period EMA
            if close[i] < ema_10[i] and close[i-1] >= ema_10[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 10-period EMA
            if close[i] > ema_10[i] and close[i-1] <= ema_10[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok and trend_ok:
                # Long: price breaks above Donchian upper band
                if close[i] > donch_high[i] and close[i-1] <= donch_high[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower band
                elif close[i] < donch_low[i] and close[i-1] >= donch_low[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals