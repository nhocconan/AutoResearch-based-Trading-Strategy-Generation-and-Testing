#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour DMI-based trend following with 4-hour ADX filter and volume confirmation.
# Uses DMI (Directional Movement Index) to identify trend direction and strength on 1h.
# Filters trades using 4h ADX > 25 to ensure strong trends only, avoiding choppy markets.
# Volume confirmation ensures institutional participation. Works in bull markets by catching
# strong uptrends and in bear markets by capturing strong downtrends while avoiding false signals
# in ranging conditions. Target: 80-150 total trades over 4 years.
name = "exp_14154_1h_dmi_adx_vol_v1"
timeframe = "1h"
leverage = 1.0

def calculate_dmi(high, low, close, period):
    """Calculate DMI (+DI, -DI, ADX) with proper min_periods"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_dm_period = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    minus_dm_period = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_period / tr_period
    minus_di = 100 * minus_dm_period / tr_period
    
    # DX and ADX
    dx = np.zeros_like(close)
    dx[period:] = 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:])
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return plus_di, minus_di, adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for ADX filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX on 4h
    _, _, adx_4h = calculate_dmi(high_4h, low_4h, close_4h, 14)
    
    # Align ADX to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate DMI on 1h
    plus_di, minus_di, adx_1h = calculate_dmi(high, low, close, 14)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    # ATR for stop loss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 14 for DMI/ADX, 20 for volume, 14 for ATR)
    start = max(14, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx_4h_aligned[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or \
           np.isnan(adx_1h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # DMI signals with 4h ADX filter and volume
        # Long: +DI > -DI and 4h ADX > 25 and volume
        # Short: -DI > +DI and 4h ADX > 25 and volume
        signal_long = (plus_di[i] > minus_di[i]) and (adx_4h_aligned[i] > 25) and vol_filter[i]
        signal_short = (minus_di[i] > plus_di[i]) and (adx_4h_aligned[i] > 25) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if signal_long:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.5 * atr[i])
            elif signal_short:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or DI crossover
            if close[i] <= stop_price or minus_di[i] > plus_di[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on stop or DI crossover
            if close[i] >= stop_price or plus_di[i] > minus_di[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals