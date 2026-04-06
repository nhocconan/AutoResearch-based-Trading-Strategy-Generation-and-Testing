#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Camarilla pivot levels for mean reversion with volume confirmation.
# Fade at R3/S3 levels (strong reversal zones) and breakout continuation at R4/S4 levels.
# Uses 1-week EMA20 as trend filter to align with higher timeframe momentum.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in ranging markets (fade at R3/S3) and trending markets (breakout at R4/S4).

name = "exp_13807_6h_camarilla1d_ema1w_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Daily pivot (uses previous day's OHLC)
TREND_EMA_PERIOD = 20  # Weekly EMA for trend filter
VOLUME_MA_PERIOD = 20  # Volume moving average
VOLUME_THRESHOLD = 1.5  # Volume must be 1.5x average
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price for pivot calculation
    pp = (high + low + close) / 3.0
    range_ = high - low
    
    # Camarilla levels
    r4 = pp + (range_ * 1.1 / 2)
    r3 = pp + (range_ * 1.1 / 4)
    r2 = pp + (range_ * 1.1 / 6)
    r1 = pp + (range_ * 1.1 / 12)
    s1 = pp - (range_ * 1.1 / 12)
    s2 = pp - (range_ * 1.1 / 6)
    s3 = pp - (range_ * 1.1 / 4)
    s4 = pp - (range_ * 1.1 / 2)
    
    return r1, r2, r3, r4, s1, s2, s3, s4, pp

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivots (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    r1, r2, r3, r4, s1, s2, s3, s4, pp = calculate_camarilla(high_1d, low_1d, close_1d)
    r1 = np.roll(r1, 1)
    r2 = np.roll(r2, 1)
    r3 = np.roll(r3, 1)
    r4 = np.roll(r4, 1)
    s1 = np.roll(s1, 1)
    s2 = np.roll(s2, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    pp = np.roll(pp, 1)
    # Set first value to NaN (no previous day)
    r1[0] = r2[0] = r3[0] = r4[0] = s1[0] = s2[0] = s3[0] = s4[0] = pp[0] = np.nan
    
    # Load weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, TREND_EMA_PERIOD)
    
    # Align daily and weekly indicators to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 6h data for entry timing and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from weekly EMA
        above_weekly_ema = close[i] > ema_1w_aligned[i]
        below_weekly_ema = close[i] < ema_1w_aligned[i]
        
        # Camarilla-based signals
        # Fade at R3/S3 (mean reversion)
        fade_long = volume_ok and close[i] <= s3_aligned[i] and close[i] > s4_aligned[i]
        fade_short = volume_ok and close[i] >= r3_aligned[i] and close[i] < r4_aligned[i]
        
        # Breakout continuation at R4/S4 (trend following)
        breakout_long = volume_ok and above_weekly_ema and close[i] > r4_aligned[i]
        breakout_short = volume_ok and below_weekly_ema and close[i] < s4_aligned[i]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
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
            # Exit long on fade signal at R3 or stop loss
            if close[i] >= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on fade signal at S3 or stop loss
            if close[i] <= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals