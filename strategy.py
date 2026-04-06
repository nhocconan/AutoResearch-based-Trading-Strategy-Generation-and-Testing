#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s strategy using 6h RSI(14) mean reversion with 1d Bollinger Bands regime filter.
# Long when RSI(14) < 30 and price touches lower Bollinger Band on 1d (oversold in range).
# Short when RSI(14) > 70 and price touches upper Bollinger Band on 1d (overbought in range).
# Uses Bollinger Band width to detect ranging markets (BW < 50th percentile of 50-period).
# Designed for 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Mean reversion works well in ranging markets which dominate BTC/ETH price action.

name = "exp_13891_6s_rsi_bbands_reversion_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
BBANDS_PERIOD = 20
BBANDS_STD = 2.0
BB_WIDTH_LOOKBACK = 50
BB_WIDTH_PERCENTILE = 50  # ranging when BB width < 50th percentile
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.2
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI with proper Wilder smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(close, period, std_dev):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower, sma

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
    
    # Load 1d data for Bollinger Bands regime filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Bollinger Bands and width
    close_1d = df_1d['close'].values
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close_1d, BBANDS_PERIOD, BBANDS_STD)
    bb_width = bb_upper - bb_lower
    
    # Calculate percentile of BB width for regime detection (ranging when width < 50th percentile)
    # Use expanding window to avoid look-ahead
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.expanding(min_periods=BB_WIDTH_LOOKBACK).quantile(BB_WIDTH_PERCENTILE/100.0).values
    ranging_market = bb_width < bb_width_percentile
    
    # Align 1d Bollinger Bands and ranging market flag to 6h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    ranging_aligned = align_htf_to_ltf(prices, df_1d, ranging_market.astype(float))
    
    # 6h data for RSI, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI for mean reversion signals
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, BBANDS_PERIOD, BB_WIDTH_LOOKBACK, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i]) or np.isnan(ranging_aligned[i]):
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
        
        # RSI mean reversion signals
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Price touching Bollinger Bands
        price_at_lower = close[i] <= bb_lower_aligned[i] * 1.001  # small tolerance
        price_at_upper = close[i] >= bb_upper_aligned[i] * 0.999  # small tolerance
        
        # Ranging market filter
        is_ranging = ranging_aligned[i] > 0.5
        
        # Mean reversion signals
        long_signal = volume_ok and rsi_oversold and price_at_lower and is_ranging
        short_signal = volume_ok and rsi_overbought and price_at_upper and is_ranging
        
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
            # Exit long on RSI > 50 (mean reversion complete) or price at middle band
            if rsi[i] > 50 or close[i] >= bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on RSI < 50 (mean reversion complete) or price at middle band
            if rsi[i] < 50 or close[i] <= bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals