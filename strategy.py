#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return signals
    
    # Calculate daily TRIX (15-period EMA applied 3 times)
    close_1d = df_1d['close'].values
    
    # TRIX calculation: EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    
    # TRIX = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix_raw = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = trix_raw.fillna(0).values
    
    # Shift by 1 to use only completed daily bars
    trix = np.roll(trix, 1)
    trix[0] = 0
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Calculate 4h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h ADX(14) for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(140, n):  # Start after warmup periods
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx[i] > 25
        
        # Long conditions: TRIX > 0 (bullish momentum) AND RSI > 50 (bullish bias) with volume and trend
        long_signal = volume_confirmed and trending and (trix_aligned[i] > 0) and (rsi[i] > 50)
        
        # Short conditions: TRIX < 0 (bearish momentum) AND RSI < 50 (bearish bias) with volume and trend
        short_signal = volume_confirmed and trending and (trix_aligned[i] < 0) and (rsi[i] < 50)
        
        # Exit when TRIX crosses zero (momentum reversal)
        exit_long = position == 1 and trix_aligned[i] <= 0
        exit_short = position == -1 and trix_aligned[i] >= 0
        
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
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: TRIX momentum + RSI bias + volume confirmation + ADX trend filter on 4h.
# Uses daily TRIX (triple EMA momentum oscillator) to capture medium-term momentum shifts.
# Enters long when daily TRIX > 0 (bullish momentum) AND 4h RSI > 50 (bullish bias)
# with volume confirmation (>1.8x average) and ADX > 25 (trending market).
# Enters short when daily TRIX < 0 (bearish momentum) AND 4h RSI < 50 (bearish bias)
# with volume confirmation and ADX > 25.
# Exits when daily TRIX crosses zero (momentum reversal).
# This combination filters out ranging markets (low ADX) and false momentum signals.
# Target: 80-150 total trades over 4 years (20-38/year) to minimize fee drag.
# TRIX is particularly effective in crypto markets for catching sustained moves.
# Works in both bull and bear markets by aligning with the dominant momentum regime.