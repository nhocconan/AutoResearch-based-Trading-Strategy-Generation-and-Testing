#!/usr/bin/env python3
"""
Strategy: 1h OHLC Breakout with 4h Trend Filter and Volume Confirmation
Timeframe: 1h
Hypothesis: Use 4h EMA for trend direction, enter on 1h breakouts above/below prior 1h high/low with volume confirmation. Works in bull (breakouts) and bear (fades) by aligning with 4h trend. Designed for low trade frequency (15-37/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14114_1h_breakout_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
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
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(20) for trend
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average (prevents low-volume noise)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for volume/ema, 14 for ATR)
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
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
        
        # Entry conditions
        # Long: break above prior 1h high with volume, 4h uptrend
        # Short: break below prior 1h low with volume, 4h downtrend
        breakout_long = (close[i] > high[i-1]) and vol_filter[i] and (ema_4h_aligned[i] > ema_4h_aligned[i-1])
        breakout_short = (close[i] < low[i-1]) and vol_filter[i] and (ema_4h_aligned[i] < ema_4h_aligned[i-1])
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_short:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reversal signal
            if close[i] <= stop_price or (close[i] < low[i-1] and vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on stop or reversal signal
            if close[i] >= stop_price or (close[i] > high[i-1] and vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals