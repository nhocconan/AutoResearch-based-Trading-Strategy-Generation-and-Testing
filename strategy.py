#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1-day Camarilla pivot levels with volume confirmation.
# Goes long when price breaks above R4 with volume (strong uptrend continuation),
# short when breaks below S4 with volume (strong downtrend continuation).
# Uses 1-week trend (EMA50) as filter to avoid counter-trend trades.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.
# Camarilla levels provide institutional support/resistance levels that work across market regimes.

name = "exp_13767_6h_camarilla1d_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Daily Camarilla (uses previous day's OHLC)
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 6
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Camarilla formula uses previous period's close
    # R4 = Close + (High - Low) * 1.1 / 2
    # R3 = Close + (High - Low) * 1.1 / 4
    # R2 = Close + (High - Low) * 1.1 / 6
    # R1 = Close + (High - Low) * 1.1 / 12
    # PP = (High + Low + Close) / 3
    # S1 = Close - (High - Low) * 1.1 / 12
    # S2 = Close - (High - Low) * 1.1 / 6
    # S3 = Close - (High - Low) * 1.1 / 4
    # S4 = Close - (High - Low) * 1.1 / 2
    
    # For daily levels, we use previous day's data
    # We'll calculate using the close of the previous bar
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # first value
    
    # For high/low, we use the same period's high/low (standard Camarilla)
    # But we need to be careful about look-ahead - we'll use rolling window
    # Actually, Camarilla uses the previous period's OHLC, not current
    
    # Simplified: use previous day's range
    prev_high = np.roll(high, 1)
    prev_high[0] = high[0]
    prev_low = np.roll(low, 1)
    prev_low[0] = low[0]
    
    range_val = prev_high - prev_low
    
    r4 = prev_close + range_val * 1.1 / 2
    r3 = prev_close + range_val * 1.1 / 4
    s3 = prev_close - range_val * 1.1 / 4
    s4 = prev_close - range_val * 1.1 / 2
    
    return r4, r3, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 1d data for Camarilla levels and 1w data for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, TREND_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r4, camarilla_r3, camarilla_s3, camarilla_s4 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 6h data for entry timing and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_ma[i]):
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
        
        # Trend direction from 1w EMA
        above_ema = close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1w_aligned[i]
        
        # Camarilla breakout signals
        long_signal = volume_ok and above_ema and close[i] > camarilla_r4_aligned[i]
        short_signal = volume_ok and below_ema and close[i] < camarilla_s4_aligned[i]
        
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
            # Exit long on close below R3 (mean reversion or trend weakening)
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above S3 (mean reversion or trend weakening)
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals