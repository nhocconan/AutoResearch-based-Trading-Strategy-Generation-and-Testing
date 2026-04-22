#!/usr/bin/env python3
"""
Hypothesis: 12-hour weekly Donchian breakout with daily ATR filter and volume confirmation.
This strategy targets breakouts from weekly price channels in trending markets, using
daily ATR to filter low-volatility conditions and volume to confirm institutional
participation. Weekly breakouts capture major trends, while daily ATR and volume filters
avoid whipsaws in ranging markets. Designed to work in both bull and bear regimes by
focusing on strong directional moves with confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Load daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily average true range as percentage of price for volatility filter
    atr_pct = atr_14_aligned / close
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_pct[i]) or np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with sufficient volatility and volume
            vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
            if (close[i] > donchian_high_aligned[i] and 
                atr_pct[i] > 0.01 and  # Minimum 1% ATR as percentage of price
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low with sufficient volatility and volume
            elif (close[i] < donchian_low_aligned[i] and 
                  atr_pct[i] > 0.01 and  # Minimum 1% ATR as percentage of price
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Donchian level or volatility drops
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to weekly Donchian low or low volatility
                if (close[i] < donchian_low_aligned[i] or atr_pct[i] < 0.005):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to weekly Donchian high or low volatility
                if (close[i] > donchian_high_aligned[i] or atr_pct[i] < 0.005):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WeeklyDonchian_Breakout_ATR_Volume"
timeframe = "12h"
leverage = 1.0