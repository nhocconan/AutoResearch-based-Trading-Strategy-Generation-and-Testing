#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week Donchian channel breakouts with volume confirmation.
# Goes long when price breaks above weekly Donchian high with volume, short when breaks below weekly low with volume.
# Uses 1-week EMA (50-period) as trend filter to avoid counter-trend trades.
# Designed for 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.
# Volatility filter: only trade when ATR ratio indicates expanding volatility (avoid chop).

name = "exp_13770_1d_donchian20_1w_ema_vol_volat"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20  # 20 weeks for weekly Donchian
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 10
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
VOLATILITY_PERIOD = 10
VOLATILITY_THRESHOLD = 1.3  # ATR ratio > 1.3 indicates expanding volatility

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data for Donchian channels and trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, TREND_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1w Donchian channels (using 20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_upper, donchian_lower = calculate_donchian(high_1w, low_1w, DONCHIAN_PERIOD)
    
    # Align Donchian levels to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # 1d data for entry timing and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss and volatility filter
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    atr_ma = pd.Series(atr).rolling(window=VOLATILITY_PERIOD, min_periods=VOLATILITY_PERIOD).mean().values
    volatility_ratio = atr / atr_ma  # Current ATR vs average ATR
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, VOLUME_MA_PERIOD, VOLATILITY_PERIOD, DONCHIAN_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(volatility_ratio[i]):
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
        
        # Volatility filter: only trade when volatility is expanding
        volatility_ok = volatility_ratio[i] > VOLATILITY_THRESHOLD
        
        # Trend direction from 1w EMA
        above_ema = close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1w_aligned[i]
        
        # Donchian breakout signals
        long_signal = volume_ok and volatility_ok and above_ema and close[i] > donchian_upper_aligned[i]
        short_signal = volume_ok and volatility_ok and below_ema and close[i] < donchian_lower_aligned[i]
        
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
            # Exit long on close below Donchian lower (mean reversion or trend change)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above Donchian upper (mean reversion or trend change)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals