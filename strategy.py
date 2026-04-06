#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot reversals with 1-day trend filter and volume confirmation.
# Camarilla levels (H4/L4 for reversal, H5/L5 for breakout) derived from prior day's range.
# Works in ranging markets (fade at H4/L4) and trending markets (breakout at H5/L5).
# 1-day EMA(50) filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume confirmation filters out low-probability setups.

name = "exp_13607_6h_camarilla_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1 / 12  # H/L = close ± (high-low)*1.1/12
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    range_ = high - low
    h5 = close + range_ * 1.1 * 2 / 2  # Actually H5 = close + range * 1.1 * 2
    h4 = close + range_ * 1.1 * 1 / 2  # H4 = close + range * 1.1 * 1
    l4 = close - range_ * 1.1 * 1 / 2  # L4 = close - range * 1.1 * 1
    l5 = close - range_ * 1.1 * 2 / 2  # L5 = close - range * 1.1 * 2
    return h4, l4, h5, l5

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
    
    # Load 1d data for Camarilla and trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    h4, l4, h5, l5 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate 1d EMA for trend filter
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    
    # Align Camarilla levels and EMA to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h5_aligned = align_htf_to_ltf(prices, df_1d, h5)
    l5_aligned = align_htf_to_ltf(prices, df_1d, l5)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
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
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) \
           or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1d EMA (price above/below EMA)
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Camarilla signals
        # Fade at H4/L4 (reversal)
        fade_long = close[i] <= l4_aligned[i] and close[i-1] > l4_aligned[i-1]  # cross above L4
        fade_short = close[i] >= h4_aligned[i] and close[i-1] < h4_aligned[i-1]  # cross below H4
        
        # Breakout at H5/L5 (continuation)
        breakout_long = close[i] >= h5_aligned[i] and close[i-1] < h5_aligned[i-1]  # break above H5
        breakdown_short = close[i] <= l5_aligned[i] and close[i-1] > l5_aligned[i-1]  # break below L5
        
        # Generate signals
        if position == 0:
            if volume_ok:
                # Fade trades (mean reversion)
                if fade_long and uptrend:  # Only long fade in uptrend
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif fade_short and downtrend:  # Only short fade in downtrend
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                # Breakout trades (trend continuation)
                elif breakout_long and uptrend:  # Breakout long in uptrend
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif breakdown_short and downtrend:  # Breakdown short in downtrend
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i))
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on fade signal at H4 or stop loss
            if close[i] >= h4_aligned[i] and close[i-1] < h4_aligned[i-1]:  # cross below H4 (exit fade)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on fade signal at L4 or stop loss
            if close[i] <= l4_aligned[i] and close[i-1] > l4_aligned[i-1]:  # cross above L4 (exit fade)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals