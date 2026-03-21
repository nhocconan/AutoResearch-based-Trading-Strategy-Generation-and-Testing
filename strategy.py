#!/usr/bin/env python3
"""
EXPERIMENT #020 - MTF KAMA Trend + RSI Pullback + Volatility Regime Filter
===========================================================================
Hypothesis: 4h KAMA (Kaufman Adaptive Moving Average) provides smoother trend
detection than HMA by adapting to market noise. Combined with 1h RSI pullback
entries and Bollinger Band Width regime filter to avoid extreme volatility periods.

Key features:
- 4h KAMA(10) trend direction (adaptive, less whipsaw than HMA)
- 1h RSI(14) pullback entries (RSI<40 long, RSI>60 short)
- BB Width percentile filter (avoid top 20% volatility)
- Explicit 2.5*ATR trailing stoploss
- Discrete position sizing (0.0, ±0.30)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_4h_kama_rsi_bbw_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    change = np.zeros(n)
    volatility = np.zeros(n)
    
    for i in range(1, n):
        change[i] = abs(close[i] - close[i-period]) if i >= period else abs(close[i] - close[0])
        vol_sum = 0.0
        for j in range(1, period+1):
            if i-j >= 0:
                vol_sum += abs(close[i-j+1] - close[i-j])
        volatility[i] = vol_sum
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constant
    sc = (er * (2.0/(period+1) - 2.0/(period+1)) + 2.0/(period+1)) ** 2
    
    # Initialize KAMA
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama


def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rsi = np.zeros(len(close))
    mask = avg_loss > 0
    rs = np.zeros(len(close))
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0  # No losses = RSI 100
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands and Band Width"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma
    
    return upper, lower, bb_width


def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h KAMA for trend
    kama_4h = calculate_kama(df_4h['close'].values, period=10)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 4h EMA for additional trend confirmation
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === CALCULATE 1h INDICATORS (vectorized before loop) ===
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Calculate BB Width percentile (rolling 100 bars)
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.percentile(x, 80), raw=True
    ).values
    
    # === GENERATE SIGNALS WITH STOPLOSS TRACKING ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    min_bars = 100  # Wait for indicators to stabilize
    
    for i in range(min_bars, n):
        # Check for NaN in any indicator
        if np.isnan(kama_4h_aligned[i]) or np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        # === TREND FILTER (4h) ===
        # Long bias: price > KAMA and KAMA rising
        trend_long = (close[i] > kama_4h_aligned[i]) and (kama_4h_aligned[i] > kama_4h_aligned[i-1])
        # Short bias: price < KAMA and KAMA falling
        trend_short = (close[i] < kama_4h_aligned[i]) and (kama_4h_aligned[i] < kama_4h_aligned[i-1])
        
        # === VOLATILITY REGIME FILTER ===
        # Avoid trading in extreme volatility (top 20% BB Width)
        high_volatility = bb_width[i] > bb_width_percentile[i] if not np.isnan(bb_width_percentile[i]) else False
        
        # === ENTRY SIGNALS (1h RSI pullback) ===
        long_entry = trend_long and (rsi_1h[i] < 40) and not high_volatility
        short_entry = trend_short and (rsi_1h[i] > 60) and not high_volatility
        
        # === STOPLOSS LOGIC ===
        stoploss_triggered = False
        
        if position_side == 1:  # Long position
            highest_price = max(highest_price, close[i])
            trail_stop = highest_price - 2.5 * atr_1h[i]
            
            if close[i] < trail_stop:
                stoploss_triggered = True
        elif position_side == -1:  # Short position
            lowest_price = min(lowest_price, close[i])
            trail_stop = lowest_price + 2.5 * atr_1h[i]
            
            if close[i] > trail_stop:
                stoploss_triggered = True
        
        # === SET SIGNAL ===
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            highest_price = 0.0
            lowest_price = 0.0
        elif long_entry and position_side != 1:
            signals[i] = SIZE
            position_side = 1
            entry_price = close[i]
            highest_price = close[i]
        elif short_entry and position_side != -1:
            signals[i] = -SIZE
            position_side = -1
            entry_price = close[i]
            lowest_price = close[i]
        elif position_side == 1:
            signals[i] = SIZE  # Hold long
        elif position_side == -1:
            signals[i] = -SIZE  # Hold short
        else:
            signals[i] = 0.0  # Flat
    
    return signals