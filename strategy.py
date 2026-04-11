#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume_trend
Strategy: 4h Donchian breakout with 1d trend filter, volume confirmation, and ATR stop
Timeframe: 4h
Leverage: 1.0
Hypothesis: Buy when price breaks above 20-period Donchian high with volume confirmation and 1d uptrend; sell when price breaks below 20-period Donchian low with volume confirmation and 1d downtrend. Exit on opposite Donchian break or ATR-based stop. Works in bull/bear markets by aligning with higher timeframe trend. Targets 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h ATR for volatility filter and stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 4h volume filter: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d Close (trend filter: use prior day's close)
    close_1d = df_1d['close'].values
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan
    close_1d_trend = align_htf_to_ltf(prices, df_1d, close_1d_shifted)
    
    # Session filter: 0-23 UTC (covers major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_30[i]) or np.isnan(close_1d_trend[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_30[i]
        atr_val = atr[i]
        
        # Volume confirmation: 4h volume must be elevated
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Trend filter: price close vs prior day close (1d trend)
        uptrend_1d = price_close > close_1d_trend[i]
        downtrend_1d = price_close < close_1d_trend[i]
        
        # Entry conditions
        long_breakout = price_high > donchian_high[i]
        short_breakout = price_low < donchian_low[i]
        
        long_signal = volume_confirmed and long_breakout and uptrend_1d
        short_signal = volume_confirmed and short_breakout and downtrend_1d
        
        # Exit conditions: opposite breakout or ATR stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on short breakout or ATR stop
            exit_long = short_breakout or (price_close < (high[i] - 2.5 * atr_val))
        elif position == -1:
            # Exit short on long breakout or ATR stop
            exit_short = long_breakout or (price_close > (low[i] + 2.5 * atr_val))
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals