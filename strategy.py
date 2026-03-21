#!/usr/bin/env python3
"""
EXPERIMENT #002 - Multi-Timeframe HMA + MACD + RSI Strategy
============================================================
Hypothesis: Combining 4h HMA trend with 1h MACD momentum and RSI pullback entries
will capture trends while avoiding late entries. The 4h HMA provides cleaner trend
direction than Supertrend, while MACD histogram confirms momentum before entry.

Key components:
- 4h HMA(21) for trend direction (HTF via mtf_data helper)
- 1h MACD(12,26,9) histogram for momentum confirmation
- 1h RSI(14) for pullback entries (buy dips in uptrend)
- ATR(14) trailing stoploss via signal→0
- Take profit: reduce to half position at 2R profit

Position sizing: 0.30 max, discrete levels (0.0, ±0.15, ±0.30)
Stoploss: 2.5*ATR against position
Take profit: 5*ATR (2R), then trail at 2*ATR
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_macd_rsi_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA calculation helper
    def wma(data, w_period):
        weights = np.arange(1, w_period + 1)
        result = np.full(len(data), np.nan)
        for i in range(w_period - 1, len(data)):
            result[i] = np.sum(data[i-w_period+1:i+1] * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, period // 2)
    wma_full = wma(close, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, int(np.sqrt(period)))
    
    return hma


def calculate_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    macd_signal = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
    macd_hist = macd_line - macd_signal
    
    return macd_line, macd_signal, macd_hist


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI with proper min_periods"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR with proper min_periods"""
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
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1) ===
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # === CALCULATE 1h INDICATORS (vectorized before loop) ===
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    # === GENERATE SIGNALS ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    HALF_SIZE = 0.15  # Half position for take profit
    
    # Track position state for stoploss/take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period for all indicators
    warmup = max(50, 26 + 9)  # MACD needs slow EMA + signal
    
    for i in range(warmup, n):
        # Check for NaN in any indicator
        if np.isnan(hma_4h_aligned[i]) or np.isnan(macd_hist[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        # === TREND FILTER: 4h HMA direction ===
        hma_slope = hma_4h_aligned[i] - hma_4h_aligned[i-1] if i > 0 else 0
        trend_bullish = hma_4h_aligned[i] > close[i] * 0.98  # Price above 4h HMA
        trend_bearish = hma_4h_aligned[i] < close[i] * 1.02  # Price below 4h HMA
        
        # === MOMENTUM: MACD histogram ===
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1]  # Rising positive
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1]  # Falling negative
        
        # === ENTRY: RSI pullback ===
        rsi_oversold = rsi[i] < 45  # Pullback in uptrend
        rsi_overbought = rsi[i] > 55  # Rally in downtrend
        
        # === STOPLOSS / TAKE PROFIT LOGIC ===
        if position_side == 1:  # Long position
            highest_since_entry = max(highest_since_entry, high[i])
            
            # Stoploss: price drops 2.5*ATR below entry
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position_side = 0
                continue
            
            # Take profit: at 5*ATR (2R), reduce to half
            if close[i] >= entry_price + 5.0 * atr[i]:
                signals[i] = HALF_SIZE
                # Trail stop: if price drops 2*ATR from highest, exit
                if close[i] < highest_since_entry - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position_side = 0
                    continue
                continue
            
            # Trail stop during normal movement
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position_side = 0
                continue
        
        elif position_side == -1:  # Short position
            lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Stoploss: price rises 2.5*ATR above entry
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position_side = 0
                continue
            
            # Take profit: at 5*ATR (2R), reduce to half
            if close[i] <= entry_price - 5.0 * atr[i]:
                signals[i] = -HALF_SIZE
                # Trail stop: if price rises 2*ATR from lowest, exit
                if close[i] > lowest_since_entry + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position_side = 0
                    continue
                continue
            
            # Trail stop during normal movement
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position_side = 0
                continue
        
        # === NEW ENTRY SIGNALS ===
        if position_side == 0:
            # Long entry: bullish trend + MACD momentum + RSI pullback
            if trend_bullish and macd_bullish and rsi_oversold:
                signals[i] = SIZE
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
            
            # Short entry: bearish trend + MACD momentum + RSI rally
            elif trend_bearish and macd_bearish and rsi_overbought:
                signals[i] = -SIZE
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
        else:
            # Maintain current position size
            if position_side == 1:
                signals[i] = SIZE if signals[i] != HALF_SIZE else HALF_SIZE
            elif position_side == -1:
                signals[i] = -SIZE if signals[i] != -HALF_SIZE else -HALF_SIZE
    
    return signals