#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d RSI divergence + 12h momentum with volume filter.
# Goes long when 1d RSI < 40 (oversold) and 12h ROC > 0 with above-average volume.
# Goes short when 1d RSI > 60 (overbought) and 12h ROC < 0 with above-average volume.
# Uses 1d ADX > 25 to filter ranging markets and avoid false signals.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# RSI divergence captures mean reversion in extremes, ROC confirms momentum direction,
# ADX filter ensures trending conditions for better signal quality.

name = "exp_13832_12h_rsi_div_mom_vol_adx_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 60
RSI_OVERSOLD = 40
ROC_PERIOD = 10
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ADX_PERIOD = 14
ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_roc(close, period):
    """Calculate Rate of Change"""
    roc = np.zeros_like(close)
    roc[period:] = (close[period:] - close[:-period]) / close[:-period] * 100
    return roc

def calculate_adx(high, low, close, period):
    """Calculate ADX"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR and DM
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

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
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for RSI and ADX filters ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI for overbought/oversold
    close_1d = df_1d['close'].values
    rsi_1d = calculate_rsi(close_1d, RSI_PERIOD)
    
    # Calculate 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d_adx, ADX_PERIOD)
    
    # Align 1d indicators to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 12h data for ROC, volume, and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ROC for momentum
    roc = calculate_roc(close, ROC_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, ROC_PERIOD, VOLUME_MA_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(roc[i]) or np.isnan(volume_ma[i]):
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
        
        # Trend filter: only trade in trending markets (ADX > 25)
        trending = adx_1d_aligned[i] > ADX_THRESHOLD
        
        # RSI and momentum conditions
        rsi_oversold = rsi_1d_aligned[i] < RSI_OVERSOLD
        rsi_overbought = rsi_1d_aligned[i] > RSI_OVERBOUGHT
        roc_positive = roc[i] > 0
        roc_negative = roc[i] < 0
        
        # Entry signals
        long_signal = volume_ok and trending and rsi_oversold and roc_positive
        short_signal = volume_ok and trending and rsi_overbought and roc_negative
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when RSI becomes overbought or ROC turns negative
            if rsi_1d_aligned[i] > RSI_OVERBOUGHT or roc[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when RSI becomes oversold or ROC turns positive
            if rsi_1d_aligned[i] < RSI_OVERSOLD or roc[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals