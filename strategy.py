#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band squeeze breakout with 4h trend filter and volume confirmation.
# In low volatility (Bollinger Band squeeze), price often breaks out strongly. 
# Use 4h EMA(50) for trend direction: only take breakouts in trend direction.
# Volume confirmation ensures institutional participation. Session filter (08-20 UTC) reduces noise.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Target: 15-37 trades/year by using strict Bollinger Band squeeze + breakout + trend + volume + session.

name = "exp_13654_1h_bb_squeeze_4h_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD_DEV = 2.0
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_bbands(close, period, std_dev):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean()
    std = pd.Series(close).rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    width = upper - lower
    return upper.values, lower.values, width.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, TREND_EMA_PERIOD)
    ema_4h_slope = np.diff(ema_4h, prepend=ema_4h[0])  # slope approximation
    ema_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slope)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_width = calculate_bbands(close, BB_PERIOD, BB_STD_DEV)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BB_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_4h_slope_aligned[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_width[i]) or np.isnan(volume_ma[i]):
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
        
        # Bollinger Band squeeze: width below 20th percentile of last 50 periods
        if i >= 50:
            width_lookback = bb_width[max(0, i-50):i]
            width_percentile = np.percentile(width_lookback, 20) if len(width_lookback) > 0 else bb_width[i]
            squeeze = bb_width[i] <= width_percentile
        else:
            squeeze = False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 4h EMA slope
        uptrend = ema_4h_slope_aligned[i] > 0
        downtrend = ema_4h_slope_aligned[i] < 0
        
        # Bollinger Band breakout signals
        # Long: price breaks above upper band during squeeze in uptrend
        long_signal = squeeze and volume_ok and in_session and uptrend and close[i] > bb_upper[i]
        
        # Short: price breaks below lower band during squeeze in downtrend
        short_signal = squeeze and volume_ok and in_session and downtrend and close[i] < bb_lower[i]
        
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
            # Exit long on opposite BB signal or stop loss
            if close[i] < bb_lower[i]:  # price breaks below lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite BB signal or stop loss
            if close[i] > bb_upper[i]:  # price breaks above upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals