#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily strategy using weekly Donchian(20) breakouts with daily RSI(2) filter and volume confirmation.
# Uses weekly trend for direction, daily RSI mean-reversion for entries, volume for confirmation.
# Designed for ~50-100 total trades over 4 years (12-25/year) to avoid excessive fees.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.
# Target: 50-100 total trades, 0.25 position size, max DD < -50%.

name = "exp_13744_1d_donchian20_wk_rsi_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters - tuned for low trade frequency
DONCHIAN_PERIOD = 20
RSI_PERIOD = 2
VOLUME_MA_PERIOD = 5
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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
    if n < 30:
        return np.zeros(n)
    
    # Load weekly data for trend filter ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    donchian_high_weekly = pd.Series(high_weekly).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low_weekly = pd.Series(low_weekly).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high_weekly)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low_weekly)
    
    # Calculate daily indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI(2) for mean-reversion signals
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i]):
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
        
        # RSI conditions for mean-reversion
        rsi_oversold = rsi[i] < 10
        rsi_overbought = rsi[i] > 90
        
        # Weekly Donchian breakout signals
        if i > 0 and not np.isnan(donchian_high_aligned[i-1]) and not np.isnan(donchian_low_aligned[i-1]):
            high_prev = donchian_high_aligned[i-1]
            low_prev = donchian_low_aligned[i-1]
            
            # Long signal: price breaks above weekly Donchian high with RSI oversold and volume
            long_signal = volume_ok and rsi_oversold and close[i] > high_prev and close[i-1] <= high_prev
            
            # Short signal: price breaks below weekly Donchian low with RSI overbought and volume
            short_signal = volume_ok and rsi_overbought and close[i] < low_prev and close[i-1] >= low_prev
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
            # Exit long on weekly Donchian break in opposite direction
            if i > 0 and not np.isnan(donchian_low_aligned[i-1]) and not np.isnan(donchian_low_aligned[i]):
                low_prev = donchian_low_aligned[i-1]
                if close[i] < low_prev and close[i-1] >= low_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = SIGNAL_SIZE
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on weekly Donchian break in opposite direction
            if i > 0 and not np.isnan(donchian_high_aligned[i-1]) and not np.isnan(donchian_high_aligned[i]):
                high_prev = donchian_high_aligned[i-1]
                if close[i] > high_prev and close[i-1] <= high_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals