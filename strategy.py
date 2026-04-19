#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX trend filter with 1d Donchian breakout and volume confirmation
# ADX > 25 indicates strong trend (avoid ranging markets)
# Donchian(20) breakout on 1d provides clear entry/exit signals
# Volume confirmation filters weak breakouts
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
# Works in both bull and bear: ADX filter avoids false signals in chop, Donchian captures trends
name = "12h_ADX_1dDonchian20_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian(20) channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels: 20-period high/low
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # 12h ADX for trend strength filter
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    # Using 14-period as standard
    period = 14
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, period * 2)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX > 25 indicates strong trend
        if adx[i] > 25:
            if position == 0:
                # Long: price breaks above Donchian high + volume confirmation
                if (close[i] > donch_high_aligned[i] and volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low + volume confirmation
                elif (close[i] < donch_low_aligned[i] and volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                    
            elif position == 1:
                # Long: exit if price breaks below Donchian low
                if close[i] < donch_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
            elif position == -1:
                # Short: exit if price breaks above Donchian high
                if close[i] > donch_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # ADX <= 25: ranging market, stay flat
            signals[i] = 0.0
            position = 0
    
    return signals