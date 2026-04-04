#!/usr/bin/env python3
"""
exp_6711_6h_elder_ray_1d_regime_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-day regime filter (ADX + EMA200).
In bull regime (price > EMA200 + ADX > 25), go long when Bear Power improves (less negative).
In bear regime (price < EMA200 + ADX > 25), go short when Bull Power deteriorates (less positive).
Uses 13-period EMA for Elder Ray calculation. Designed to capture trending moves while
avoiding chop via ADX filter. Works in both bull and bear markets by adapting to regime.
Target: 75-150 total trades over 4 years (19-37/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6711_6h_elder_ray_1d_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_LEN = 13          # For Elder Ray
EMA200_LEN = 200      # Regime filter
ADX_LEN = 14
ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8     # ~2 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for regime filters
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA200 and ADX for regime
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA200
    ema200 = pd.Series(close_1d).ewm(span=EMA200_LEN, adjust=False).mean().values
    
    # ADX calculation
    # +DM, -DM, TR
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr3 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr_1d = pd.Series(tr).ewm(span=ADX_LEN, adjust=False).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=ADX_LEN, adjust=False).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=ADX_LEN, adjust=False).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=ADX_LEN, adjust=False).mean().values
    
    # Align HTF regime indicators to LTF (6h)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray components (13-period EMA)
    ema13 = pd.Series(close).ewm(span=EMA_LEN, adjust=False).mean().values
    bull_power = high - ema13          # Bull Power: High - EMA13
    bear_power = low - ema13           # Bear Power: Low - EMA13
    
    # Volume filter (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1_ltf = pd.Series(high - low)
    tr2_ltf = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3_ltf = pd.Series(np.abs(low - np.roll(close, 1)))
    tr_ltf = pd.concat([tr1_ltf, tr2_ltf, tr3_ltf], axis=1).max(axis=1)
    atr = tr_ltf.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(EMA200_LEN, ADX_LEN, EMA_LEN, 20) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(ema200_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine 1-day regime
        bull_regime = (close[i] > ema200_aligned[i]) and (adx_1d_aligned[i] > ADX_THRESHOLD)
        bear_regime = (close[i] < ema200_aligned[i]) and (adx_1d_aligned[i] > ADX_THRESHOLD)
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * 1.5 if not np.isnan(vol_ma[i]) else False
        
        # Elder Ray signals
        # In bull regime: look for improving Bear Power (less negative = weakening bears)
        long_signal = bull_regime and vol_confirmed and (bear_power[i] > bear_power[i-1])
        
        # In bear regime: look for deteriorating Bull Power (less positive = weakening bulls)
        short_signal = bear_regime and vol_confirmed and (bull_power[i] < bull_power[i-1])
        
        # Exit conditions: reverse signals or power crosses zero
        exit_long = position == 1 and (bear_power[i] >= 0 or not bull_regime)
        exit_short = position == -1 and (bull_power[i] <= 0 or not bear_regime)
        
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Manage existing position
            if (position == 1 and exit_long) or (position == -1 and exit_short):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = position * SIGNAL_SIZE
    
    return signals