#!/usr/bin/env python3
"""
Experiment #8611: 6h Camarilla pivot fade/breakout + 1d trend filter + volume confirmation.
Hypothesis: Camarilla levels from daily timeframe provide institutional support/resistance.
Fade at R3/S3 (mean reversion in range), breakout at R4/S4 (trend continuation).
Uses 1d EMA50 for trend filter to avoid counter-trend trades. Volume confirms institutional participation.
Targets 75-150 total trades over 4 years (19-38/year) to balance frequency and edge.
Works in bull/bear via trend filter and adaptive Camarilla logic.
"""

from mtf_data import get_ath_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8611_6h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1
TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
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

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_ = high - low
    if range_ <= 0:
        return close, close, close, close, close, close, close, close
    close_prev = close
    # Camarilla levels
    S1 = close_prev - (range_ * 1.0 / 12)
    S2 = close_prev - (range_ * 2.0 / 12)
    S3 = close_prev - (range_ * 3.0 / 12)
    S4 = close_prev - (range_ * 4.0 / 12)
    R1 = close_prev + (range_ * 1.0 / 12)
    R2 = close_prev + (range_ * 2.0 / 12)
    R3 = close_prev + (range_ * 3.0 / 12)
    R4 = close_prev + (range_ * 4.0 / 12)
    return S1, S2, S3, S4, R1, R2, R3, R4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_PERIOD, adjust=False, min_periods=TREND_PERIOD).mean().values
    
    # Price relative to 1d EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, 
                     np.where(close_1d < ema_1d, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    camarilla_levels = []
    for i in range(len(high_1d)):
        s1, s2, s3, s4, r1, r2, r3, r4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d_arr[i])
        camarilla_levels.append([s1, s2, s3, s4, r1, r2, r3, r4])
    
    camarilla_array = np.array(camarilla_levels)
    s1_1d = camarilla_array[:, 0]
    s2_1d = camarilla_array[:, 1]
    s3_1d = camarilla_array[:, 2]
    s4_1d = camarilla_array[:, 3]
    r1_1d = camarilla_array[:, 4]
    r2_1d = camarilla_array[:, 5]
    r3_1d = camarilla_array[:, 6]
    r4_1d = camarilla_array[:, 7]
    
    # Align Camarilla levels to 6h timeframe
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(s3_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d price below EMA50
        
        # Volume confirmation
        volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla-based entry logic
        # Fade at S3/R3 (mean reversion) when price touches these levels
        # Breakout at S4/R4 (trend continuation) when price breaks these levels
        
        # Long conditions
        long_fade = (bull_bias or not bear_bias) and \
                   close[i] <= s3_1d_aligned[i] * (1 + CAMARILLA_MULT/100) and \
                   close[i] >= s3_1d_aligned[i] * (1 - CAMARILLA_MULT/100) and \
                   volume_confirmed
        
        long_breakout = bull_bias and \
                       close[i] > r4_1d_aligned[i] and \
                       volume_confirmed
        
        # Short conditions
        short_fade = (bear_bias or not bull_bias) and \
                    close[i] >= r3_1d_aligned[i] * (1 - CAMARILLA_MULT/100) and \
                    close[i] <= r3_1d_aligned[i] * (1 + CAMARILLA_MULT/100) and \
                    volume_confirmed
        
        short_breakout = bear_bias and \
                        close[i] < s4_1d_aligned[i] and \
                        volume_confirmed
        
        long_entry = long_fade or long_breakout
        short_entry = short_fade or short_breakout
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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