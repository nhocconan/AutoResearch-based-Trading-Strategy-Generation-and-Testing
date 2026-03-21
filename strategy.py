#!/usr/bin/env python3
"""
EXPERIMENT #010 - Regime-Based HMA + Bollinger + RSI Strategy (4h)
==================================================================
Hypothesis: Combining HTF trend regime (1d HMA) with LTF volatility regime 
(4h Bollinger Band Width) and momentum entry (4h RSI) will outperform pure 
trend-following by avoiding choppy markets and entering on pullbacks within 
established trends.

Key components:
1. 1d HMA(21) - Long-term trend regime filter (only trade with HTF trend)
2. 4h BB Width percentile - Volatility regime (avoid squeeze, trade expansion)
3. 4h RSI(14) - Entry timing on pullbacks within trend
4. ATR(14) stoploss - Dynamic risk management at 2*ATR

Position sizing: 0.30 (30% of capital), discrete levels
Stoploss: Signal → 0 when price moves 2*ATR against position
Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_hma_bb_rsi_4h_v1"
timeframe = "4h"
leverage = 1.0


def calculate_hma(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    
    return hma.values


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> tuple:
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bw = (upper - lower) / sma  # Band width as % of price
    
    return upper.values, lower.values, bw.values


def calculate_bb_percentile(bw: np.ndarray, lookback: int = 100) -> np.ndarray:
    """Calculate BB Width percentile rank over lookback period"""
    n = len(bw)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = bw[i-lookback:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid <= bw[i]) / len(valid)
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE (1d for trend regime) ===
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === CALCULATE 4h INDICATORS ===
    hma_4h = calculate_hma(close, 21)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bb_percentile(bb_width, 100)
    
    # === GENERATE SIGNALS ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    
    first_valid = max(100, 21)  # Wait for all indicators to warm up
    
    for i in range(first_valid, n):
        # Skip if any indicator is NaN
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h[i]) or 
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_percentile[i])):
            signals[i] = 0.0
            position_side = 0
            continue
        
        # === HTF TREND REGIME (1d HMA slope) ===
        hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-1] if i > 0 else 0
        long_regime = hma_1d_slope > 0  # 1d trend is up
        short_regime = hma_1d_slope < 0  # 1d trend is down
        
        # === VOLATILITY REGIME (BB Width percentile) ===
        # Only trade when BB width is expanding (percentile > 0.5)
        vol_expansion = bb_percentile[i] > 0.50
        
        # === ENTRY SIGNALS ===
        # Long: HTF uptrend + vol expansion + RSI pullback (30-50)
        long_signal = (long_regime and vol_expansion and 
                       30 <= rsi[i] <= 50 and 
                       close[i] > hma_4h[i])
        
        # Short: HTF downtrend + vol expansion + RSI rally (50-70)
        short_signal = (short_regime and vol_expansion and 
                        50 <= rsi[i] <= 70 and 
                        close[i] < hma_4h[i])
        
        # === STOPLOSS LOGIC (2*ATR) ===
        if position_side == 1:  # Long position
            highest_since_entry = max(highest_since_entry, close[i])
            stop_loss = entry_price - 2.0 * atr[i]
            trail_stop = highest_since_entry - 2.0 * atr[i]
            effective_stop = max(stop_loss, trail_stop)
            
            if close[i] < effective_stop:
                signals[i] = 0.0
                position_side = 0
                continue
        
        elif position_side == -1:  # Short position
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_loss = entry_price + 2.0 * atr[i]
            trail_stop = lowest_since_entry + 2.0 * atr[i]
            effective_stop = min(stop_loss, trail_stop)
            
            if close[i] > effective_stop:
                signals[i] = 0.0
                position_side = 0
                continue
        
        # === TAKE PROFIT (reduce at 2R) ===
        if position_side == 1:
            profit = close[i] - entry_price
            risk = entry_price - (entry_price - 2.0 * atr[entry_idx]) if 'entry_idx' in dir() else atr[i]
            if profit >= 2.0 * atr[i] and signals[i-1] == SIZE:
                signals[i] = SIZE / 2  # Reduce to half
                continue
        
        elif position_side == -1:
            profit = entry_price - close[i]
            if profit >= 2.0 * atr[i] and signals[i-1] == -SIZE:
                signals[i] = -SIZE / 2  # Reduce to half
                continue
        
        # === GENERATE SIGNAL ===
        if long_signal and position_side != 1:
            signals[i] = SIZE
            position_side = 1
            entry_price = close[i]
            highest_since_entry = close[i]
            entry_idx = i
        
        elif short_signal and position_side != -1:
            signals[i] = -SIZE
            position_side = -1
            entry_price = close[i]
            lowest_since_entry = close[i]
            entry_idx = i
        
        elif not long_signal and not short_signal:
            # Hold existing position or stay flat
            if position_side == 0:
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]  # Hold position
    
    return signals