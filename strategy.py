#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily volatility breakout with volume confirmation and regime filter.
# Uses ATR-based breakout from daily open to capture momentum in both bull and bear markets.
# Enters long when price breaks above daily open + k*ATR with volume surge, short when breaks below.
# Filters trades using daily ADX to avoid choppy markets. Target: 25-40 trades/year to minimize fee drag.

name = "4h_DailyVolatilityBreakout_ADXFilter"
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
    
    # Get daily data for volatility breakout and regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility breakout
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily open price for breakout levels
    open_1d = df_1d['open'].values
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    
    # Calculate ADX for regime filter (trending vs ranging)
    # +DM and -DM calculation
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume surge filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for ATR (14) and ADX (14+14)
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(open_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_surge[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_val = atr_1d_aligned[i]
        daily_open = open_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        vol_surge = volume_surge[i]
        
        # Only trade in trending markets (ADX > 25)
        if adx_val > 25:
            if position == 0:
                # Enter long: Price breaks above daily open + 0.5*ATR with volume surge
                if close[i] > daily_open + 0.5 * atr_val and vol_surge:
                    signals[i] = 0.25
                    position = 1
                # Enter short: Price breaks below daily open - 0.5*ATR with volume surge
                elif close[i] < daily_open - 0.5 * atr_val and vol_surge:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                # Exit long: Price returns to daily open or ADX drops
                if close[i] < daily_open or adx_val < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:
                # Exit short: Price returns to daily open or ADX drops
                if close[i] > daily_open or adx_val < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In ranging markets, flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals