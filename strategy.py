#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour price action filtered by 1-day volatility regime and volume confirmation.
# Uses Bollinger Band width percentile to detect low volatility (squeeze) conditions on daily timeframe.
# Enters on 6h Donchian breakout in direction of daily trend when volatility is low (pre-expansion).
# Includes volume confirmation to ensure institutional participation and session filter (08-20 UTC).
# Designed to work in both bull and bear markets by capturing volatility breakouts after consolidation.
# Target: 20-40 trades/year by combining low-frequency volatility regime filter with precise entry timing.

name = "exp_13631_6h_bbw_regime_donchian_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
BB_LENGTH = 20
BB_STD = 2.0
BBW_PERCENTILE_LOOKBACK = 50
BBW_PERCENTILE_THRESHOLD = 30  # Below 30th percentile = low volatility (squeeze)
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_bbands(close, length, std):
    """Calculate Bollinger Bands"""
    basis = pd.Series(close).rolling(window=length, min_periods=length).mean().values
    dev = pd.Series(close).rolling(window=length, min_periods=length).std().values
    upper = basis + (dev * std)
    lower = basis - (dev * std)
    return basis, upper, lower

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for volatility regime and trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Bollinger Band width for volatility regime
    close_1d = df_1d['close'].values
    _, bb_upper_1d, bb_lower_1d = calculate_bbands(close_1d, BB_LENGTH, BB_STD)
    bb_width_1d = (bb_upper_1d - bb_lower_1d) / np.where(bb_upper_1d + bb_lower_1d == 0, 1e-10, bb_upper_1d + bb_lower_1d) * 2  # Normalize
    # Calculate percentile of BB width (lower = more squeezed)
    bb_width_series = pd.Series(bb_width_1d)
    bb_width_percentile = bb_width_series.rolling(window=BBW_PERCENTILE_LOOKBACK, min_periods=BBW_PERCENTILE_LOOKBACK).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Calculate 1d trend (using close vs 50-period EMA)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = close_1d > ema_50_1d  # True for uptrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d.astype(float))
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BBW_PERCENTILE_LOOKBACK, DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(bb_width_percentile_aligned[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
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
        
        # Volatility regime: low volatility (squeeze) condition
        low_volatility = bb_width_percentile_aligned[i] < BBW_PERCENTILE_THRESHOLD
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Breakout conditions
        breakout_up = close[i] > donch_upper[i]
        breakout_down = close[i] < donch_lower[i]
        
        # Trend alignment
        uptrend = trend_1d_aligned[i] > 0.5
        downtrend = trend_1d_aligned[i] <= 0.5
        
        # Entry signals: breakout in direction of daily trend during low volatility
        long_signal = low_volatility and volume_ok and breakout_up and uptrend and in_session
        short_signal = low_volatility and volume_ok and breakout_down and downtrend and in_session
        
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
            # Exit long on opposite breakout or stop loss
            if breakout_down:  # Opposite breakout
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite breakout or stop loss
            if breakout_up:  # Opposite breakout
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals