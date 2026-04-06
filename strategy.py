#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot reversal with 1-day trend filter and volume confirmation.
# Uses 1-day EMA(50) for trend direction, Camarilla levels from 1d for reversal entries,
# and volume spike for confirmation. Designed for 100-180 total trades over 4 years.
# Works in bull (fade at S3/S4 in uptrend) and bear (fade at R3/R4 in downtrend) markets.
# Target: 120 total trades, 0.25 position size, max DD < -50%.

name = "exp_13711_6h_camarilla1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 10
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Load 1d data for trend filter and Camarilla levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).fillna(method='bfill').values  # Previous day close
    
    # Camarilla levels: H4/L4, H3/L3, H2/L2, H1/L1
    # H4 = Close + 1.1*(High-Low)*1.1/2
    # L4 = Close - 1.1*(High-Low)*1.1/2
    # H3 = Close + 1.1*(High-Low)*1.1/4
    # L3 = Close - 1.1*(High-Low)*1.1/4
    range_1d = high_1d - low_1d
    h4 = close_1d_prev + CAMARILLA_MULT * range_1d * 1.1 / 2
    l4 = close_1d_prev - CAMARILLA_MULT * range_1d * 1.1 / 2
    h3 = close_1d_prev + CAMARILLA_MULT * range_1d * 1.1 / 4
    l3 = close_1d_prev - CAMARILLA_MULT * range_1d * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(volume_ma[i]):
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
        
        # Trend direction from 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Camarilla reversal signals
        # Long: price rejects at L3/L4 in uptrend (bounce from support)
        long_signal = volume_ok and above_ema and \
                    ((low[i] <= l3_aligned[i] and close[i] > l3_aligned[i]) or \
                     (low[i] <= l4_aligned[i] and close[i] > l4_aligned[i]))
        
        # Short: price rejects at H3/H4 in downtrend (rejection from resistance)
        short_signal = volume_ok and below_ema and \
                     ((high[i] >= h3_aligned[i] and close[i] < h3_aligned[i]) or \
                      (high[i] >= h4_aligned[i] and close[i] < h4_aligned[i]))
        
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
            # Exit long on stop or when price crosses above H3 (failure of bounce)
            if close[i] >= h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on stop or when price crosses below L3 (failure of rejection)
            if close[i] <= l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals