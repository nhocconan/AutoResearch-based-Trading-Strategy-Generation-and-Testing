#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour strategy combining Donchian(20) breakouts with 1-week EMA(50) trend filter and volume confirmation.
# Uses 1-week trend for directional bias (avoids counter-trend trades), 4h Donchian breakouts for entries,
# and volume surge for confirmation. Designed for ~100-180 total trades over 4 years (25-45/year)
# to balance opportunity with fee minimization. Works in bull markets (breakouts with volume) and
# bear markets (breakdowns with volume) by only trading in direction of higher timeframe trend.
# Target: 100-180 total trades, 0.25 position size, max DD < -50%.

name = "exp_13757_4h_donchian20_1w_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters - tuned for low-moderate trade frequency
DONCHIAN_PERIOD = 20
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 4h and 1w data for filters ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, TREND_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 4h indicators for entries
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # ATR for stop loss (using 4h data)
    high = high_4h
    low = low_4h
    close = close_4h
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # 4h Donchian channels
    donchian_high = pd.Series(high_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for 4h
    volume_ma_4h = pd.Series(volume_4h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma_4h[i]):
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
        
        # Volume confirmation (using 4h volume)
        volume_ok = volume_4h[i] > (volume_ma_4h[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1w EMA
        above_ema = close_4h[i] > ema_1w_aligned[i]
        below_ema = close_4h[i] < ema_1w_aligned[i]
        
        # Donchian breakout signals
        if i > 0 and not np.isnan(donchian_high[i-1]) and not np.isnan(donchian_low[i-1]):
            high_prev = donchian_high[i-1]
            low_prev = donchian_low[i-1]
            
            # Long signal: price breaks above Donchian high in uptrend
            long_signal = volume_ok and above_ema and close_4h[i] > high_prev and close_4h[i-1] <= high_prev
            
            # Short signal: price breaks below Donchian low in downtrend
            short_signal = volume_ok and below_ema and close_4h[i] < low_prev and close_4h[i-1] >= low_prev
        else:
            long_signal = False
            short_signal = False
        
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
            # Exit long on opposite Donchian break
            if i > 0 and not np.isnan(donchian_low[i-1]) and not np.isnan(donchian_low[i]):
                low_prev = donchian_low[i-1]
                if close_4h[i] < low_prev and close_4h[i-1] >= low_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = SIGNAL_SIZE
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite Donchian break
            if i > 0 and not np.isnan(donchian_high[i-1]) and not np.isnan(donchian_high[i]):
                high_prev = donchian_high[i-1]
                if close_4h[i] > high_prev and close_4h[i-1] <= high_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals