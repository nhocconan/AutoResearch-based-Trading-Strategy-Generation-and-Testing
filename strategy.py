#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with daily volume confirmation and ADX filter
# Works in bull/bear because breakouts capture strong moves, volume filters weak signals,
# and ADX ensures we only trade when trending. Target: 50-150 trades over 4 years.

name = "exp_12882_12h_donchian20_1d_vol_adx_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ADX_PERIOD = 14
ADX_THRESHOLD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / \
              pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / \
               pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on daily
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    
    # Calculate upper and lower bands
    upper_d = pd.Series(high_d).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lower_d = pd.Series(low_d).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Align to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_daily, upper_d)
    lower_aligned = align_htf_to_ltf(prices, df_daily, lower_d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Donchian levels not available
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume and ADX confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        adx_ok = adx[i] > ADX_THRESHOLD if not np.isnan(adx[i]) else False
        
        # Breakout above upper or below lower band
        breakout_long = volume_ok and adx_ok and close[i] >= upper_aligned[i]
        breakout_short = volume_ok and adx_ok and close[i] <= lower_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals