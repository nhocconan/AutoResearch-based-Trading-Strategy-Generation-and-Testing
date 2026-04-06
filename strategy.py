#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h ADX trend filter and volume confirmation.
# In ranging markets (ADX<25), price reverts to VWAP; in trending markets (ADX>25), 
# we avoid trades to prevent whipsaw. Volume confirms genuine moves.
# Works in bull/bear because mean reversion works in ranges, trend filter avoids losses in trends.
# Target: 80-150 total trades over 4 years (20-38/year) with low turnover.

name = "exp_13054_1h_vwap_meanrev_4h_adx_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
VWAP_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX"""
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

def calculate_vwap(high, low, close, volume, period):
    """Calculate VWAP"""
    typical_price = (high + low + close) / 3
    vwap_num = (typical_price * volume).rolling(window=period, min_periods=period).sum()
    vwap_den = volume.rolling(window=period, min_periods=period).sum()
    vwap = (vwap_num / vwap_den).values
    return vwap

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h ADX for trend filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, ADX_PERIOD)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP
    vwap = calculate_vwap(high, low, close, volume, VWAP_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VWAP_PERIOD, ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if ADX not available
        if np.isnan(adx_4h_aligned[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: only trade when ADX < 25 (ranging market)
        ranging = adx_4h_aligned[i] < ADX_THRESHOLD
        
        # Mean reversion signals
        deviation = (close[i] - vwap[i]) / vwap[i] if vwap[i] != 0 else 0
        mean_revert_long = deviation < -0.008  # 0.8% below VWAP
        mean_revert_short = deviation > 0.008   # 0.8% above VWAP
        
        # Generate signals
        if position == 0:
            if volume_ok and ranging:
                if mean_revert_long:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif mean_revert_short:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals