#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h volume-weighted VWAP deviation with 1d trend filter.
# Buying when price deviates below VWAP in uptrend (mean reversion within trend),
# selling when price deviates above VWAP in downtrend. Uses volume confirmation
# to avoid false signals. Works in both bull (buy dips) and bear (sell rallies).

name = "exp_13607_6h_vwap_dev_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VWAP_PERIOD = 20
TREND_EMA_PERIOD = 50
DEV_THRESHOLD = 0.015  # 1.5% deviation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_vwap(high, low, close, volume, period):
    """Calculate VWAP over period"""
    typical_price = (high + low + close) / 3.0
    vwap_num = np.convolve(typical_price * volume, np.ones(period), 'valid')
    vwap_den = np.convolve(volume, np.ones(period), 'valid')
    vwap = np.full_like(typical_price, np.nan)
    vwap[period-1:] = vwap_num / vwap_den
    return vwap

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_slope = np.diff(ema_1d, prepend=ema_1d[0])  # slope approximation
    ema_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slope)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP
    vwap = calculate_vwap(high, low, close, volume, VWAP_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VWAP_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(vwap[i]) or np.isnan(ema_1d_slope_aligned[i]) or np.isnan(volume_ma[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1d EMA slope
        uptrend = ema_1d_slope_aligned[i] > 0
        downtrend = ema_1d_slope_aligned[i] < 0
        
        # VWAP deviation
        if vwap[i] != 0:
            vwap_dev = (close[i] - vwap[i]) / vwap[i]
        else:
            vwap_dev = 0
        
        # Entry conditions
        # Long: price below VWAP (oversold) in uptrend
        long_signal = volume_ok and uptrend and (vwap_dev < -DEV_THRESHOLD)
        # Short: price above VWAP (overbought) in downtrend
        short_signal = volume_ok and downtrend and (vwap_dev > DEV_THRESHOLD)
        
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
            # Exit long on mean reversion or stop loss
            if vwap_dev > 0:  # price crossed back above VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on mean reversion or stop loss
            if vwap_dev < 0:  # price crossed back below VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals