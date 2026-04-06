#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14007_6d_trix15_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_trix(close, period):
    """Calculate TRIX: triple EMA of percent change"""
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    # Calculate percent change of triple EMA
    roc = ema3.pct_change(1)
    # Signal line is EMA of ROC
    trix = roc.ewm(span=period, adjust=False, min_periods=period).mean() * 100
    return trix.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (alpha = 1/period)
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / tr_smooth
    minus_di = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for TRIX and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d TRIX(15) for momentum
    trix_1d = calculate_trix(df_1d['close'].values, 15)
    
    # Calculate 1d ADX(14) for trend strength
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align 1d indicators to 6h timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h data for volume and price
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(15, 14, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(trix_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or \
           np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        
        # Momentum signal from TRIX (crossing zero line)
        trix_cross_up = trix_1d_aligned[i] > 0 and trix_1d_aligned[i-1] <= 0
        trix_cross_down = trix_1d_aligned[i] < 0 and trix_1d_aligned[i-1] >= 0
        
        # Trend strength filter (ADX > 20)
        strong_trend = adx_1d_aligned[i] > 20
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Entry signals
        long_signal = trix_cross_up and strong_trend and volume_ok
        short_signal = trix_cross_down and strong_trend and volume_ok
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr_1d[i]) if 'atr_1d' in locals() else entry_price * 0.97
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr_1d[i]) if 'atr_1d' in locals() else entry_price * 1.03
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on TRIX cross down or weak trend
            if trix_cross_down or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on TRIX cross up or weak trend
            if trix_cross_up or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Fix: calculate ATR for stop loss using 1d data
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for TRIX, ADX, and ATR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d TRIX(15) for momentum
    trix_1d = calculate_trix(df_1d['close'].values, 15)
    
    # Calculate 1d ADX(14) for trend strength
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Calculate 1d ATR(14) for stop loss
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align 1d indicators to 6h timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6h data for volume and price
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(15, 14, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(trix_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        
        # Momentum signal from TRIX (crossing zero line)
        trix_cross_up = trix_1d_aligned[i] > 0 and trix_1d_aligned[i-1] <= 0
        trix_cross_down = trix_1d_aligned[i] < 0 and trix_1d_aligned[i-1] >= 0
        
        # Trend strength filter (ADX > 20)
        strong_trend = adx_1d_aligned[i] > 20
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Entry signals
        long_signal = trix_cross_up and strong_trend and volume_ok
        short_signal = trix_cross_down and strong_trend and volume_ok
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr_1d_aligned[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr_1d_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on TRIX cross down or weak trend
            if trix_cross_down or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on TRIX cross up or weak trend
            if trix_cross_up or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Final version with corrected variable scope
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for TRIX, ADX, and ATR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d TRIX(15) for momentum
    trix_1d = calculate_trix(df_1d['close'].values, 15)
    
    # Calculate 1d ADX(14) for trend strength
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Calculate 1d ATR(14) for stop loss
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align 1d indicators to 6h timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6h data for volume and price
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(15, 14, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(trix_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        
        # Momentum signal from TRIX (crossing zero line)
        trix_cross_up = trix_1d_aligned[i] > 0 and trix_1d_aligned[i-1] <= 0
        trix_cross_down = trix_1d_aligned[i] < 0 and trix_1d_aligned[i-1] >= 0
        
        # Trend strength filter (ADX > 20)
        strong_trend = adx_1d_aligned[i] > 20
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Entry signals
        long_signal = trix_cross_up and strong_trend and volume_ok
        short_signal = trix_cross_down and strong_trend and volume_ok
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr_1d_aligned[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr_1d_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on TRIX cross down or weak trend
            if trix_cross_down or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on TRIX cross up or weak trend
            if trix_cross_up or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Clean final version
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for TRIX, ADX, and ATR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d TRIX(15) for momentum
    trix_1d = calculate_trix(df_1d['close'].values, 15)
    
    # Calculate 1d ADX(14) for trend strength
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Calculate 1d ATR(14) for stop loss
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align 1d indicators to 6h timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6h data for volume and price
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(15, 14, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(trix_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        
        # Momentum signal from TRIX (crossing zero line)
        trix_cross_up = trix_1d_aligned[i] > 0 and trix_1d_aligned[i-1] <= 0
        trix_cross_down = trix_1d_aligned[i] < 0 and trix_1d_aligned[i-1] >= 0
        
        # Trend strength filter (ADX > 20)
        strong_trend = adx_1d_aligned[i] > 20
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Entry signals
        long_signal = trix_cross_up and strong_trend and volume_ok
        short_signal = trix_cross_down and strong_trend and volume_ok
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr_1d_aligned[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr_1d_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on TRIX cross down or weak trend
            if trix_cross_down or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on TRIX cross up or weak trend
            if trix_cross_up or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX(15) on daily timeframe captures medium-term momentum shifts. 
# Combined with ADX>20 (trending market) and volume spikes, this filters false signals.
# Works in both bull/bear markets by following momentum direction.
# Target: 50-150 trades over 4 years (12-37/year) with 0.25 position size.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for TRIX, ADX, and ATR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d TRIX(15) for momentum
    trix_1d = calculate_trix(df_1d['close'].values, 15)
    
    # Calculate 1d ADX(14) for trend strength
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Calculate 1d ATR(14) for stop loss
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align 1d indicators to 6h timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6h data for volume and price
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(15, 14, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(trix_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        
        # Momentum signal from TRIX (crossing zero line)
        trix_cross_up = trix_1d_aligned[i] > 0 and trix_1d_aligned[i-1] <= 0
        trix_cross_down = trix_1d_aligned[i] < 0 and trix_1d_aligned[i-1] >= 0
        
        # Trend strength filter (ADX > 20)
        strong_trend = adx_1d_aligned[i] > 20
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Entry signals
        long_signal = trix_cross_up and strong_trend and volume_ok
        short_signal = trix_cross_down and strong_trend and volume_ok
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr_1d_aligned[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr_1d_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on TRIX cross down or weak trend
            if trix_cross_down or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on TRIX cross up or weak trend
            if trix_cross_up or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
)"), "calculate_trix": calculate_trix, "calculate_adx": calculate_adx, "calculate_atr": calculate_atr}

# Final clean implementation
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for TRIX, ADX, and ATR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d TRIX(15) for momentum
    trix_1d = calculate_trix(df_1d['close'].values, 15)
    
    # Calculate 1d ADX(14) for trend strength
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Calculate 1d ATR(14) for stop loss
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align 1d indicators to 6h timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6h data for volume and price
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(15, 14, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(trix_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        
        # Momentum signal from TRIX (crossing zero line)
        trix_cross_up = trix_1d_aligned[i] > 0 and trix_1d_aligned[i-1] <= 0
        trix_cross_down = trix_1d_aligned[i] < 0 and trix_1d_aligned[i-1] >= 0
        
        # Trend strength filter (ADX > 20)
        strong_trend = adx_1d_aligned[i] > 20
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Entry signals
        long_signal = trix_cross_up and strong_trend and volume_ok
        short_signal = trix_cross_down and strong_trend and volume_ok
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr_1d_aligned[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr_1d_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on TRIX cross down or weak trend
            if trix_cross_down or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on TRIX cross up or weak trend
            if trix_cross_up or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
), "calculate_trix": calculate_trix, "calculate_adx": calculate_adx, "calculate_atr": calculate_atr}

# Final submission
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for TRIX, ADX, and ATR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d TRIX(15) for momentum
    trix_1d = calculate_trix(df_1d['close'].values, 15)
    
    # Calculate 1d ADX(14) for trend strength
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Calculate 1d ATR(14) for stop loss
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align 1d indicators to 6h timeframe
    trix