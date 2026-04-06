#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot reversal strategy with daily volume confirmation and weekly trend filter.
# Uses Camarilla levels (H3/L3) calculated from prior day's range to identify reversal zones in ranging markets.
# Volume confirmation filters out false breaks, weekly EMA ensures alignment with higher timeframe momentum.
# Works in sideways markets (reversions at H3/L3) and trending markets (breaks through H4/L4 with volume).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13502_12h_camarilla_1d_vol_1w_ema_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
EMA_PERIOD = 21
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Load daily data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    # We use H3/L3 for reversals and H4/L4 for breakouts
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_h4 = prev_close + CAMARILLA_MULT * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - CAMARILLA_MULT * (prev_high - prev_low) / 2
    camarilla_h3 = prev_close + CAMARILLA_MULT * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - CAMARILLA_MULT * (prev_high - prev_low) / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(volume_ma[i])):
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
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Camarilla-based signals
        # Long setup: price at or below L3 with volume and uptrend bias
        long_setup = volume_ok and uptrend and (low[i] <= camarilla_l3_aligned[i])
        # Short setup: price at or above H3 with volume and downtrend bias
        short_setup = volume_ok and downtrend and (high[i] >= camarilla_h3_aligned[i])
        
        # Breakout escapes: price breaks H4/L4 with volume (strong momentum)
        breakout_up = volume_ok and (high[i] > camarilla_h4_aligned[i])
        breakout_down = volume_ok and (low[i] < camarilla_l4_aligned[i])
        
        # Generate signals
        if position == 0:
            if long_setup:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_setup:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_up:
                # Strong upside breakout - go long
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
                # Strong downside breakout - go short
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long if price reaches H3 (take profit) or breaks L4 (stop and reverse)
            if high[i] >= camarilla_h3_aligned[i]:
                signals[i] = 0.0  # Take profit at H3
                position = 0
            elif low[i] < camarilla_l4_aligned[i]:
                signals[i] = -SIGNAL_SIZE  # Reverse to short
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short if price reaches L3 (take profit) or breaks H4 (stop and reverse)
            if low[i] <= camarilla_l3_aligned[i]:
                signals[i] = 0.0  # Take profit at L3
                position = 0
            elif high[i] > camarilla_h4_aligned[i]:
                signals[i] = SIGNAL_SIZE  # Reverse to long
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals