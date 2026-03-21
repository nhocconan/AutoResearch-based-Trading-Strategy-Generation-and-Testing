#!/usr/bin/env python3
"""
EXPERIMENT #001 - 15m RSI Pullback with 4h HMA Trend Filter
============================================================
Hypothesis: Using 4h HMA for trend direction + 15m RSI for pullback entries 
will capture trends with better entry timing than pure Supertrend. The HTF
filter reduces whipsaws while LTF RSI finds optimal entry points during 
pullbacks. ATR-based stoploss controls downside risk.

Key features:
- 4h HMA(21) for trend direction (HTF filter) - call ONCE before loop
- 15m RSI(14) for entry timing (oversold in uptrend, overbought in downtrend)
- ATR(14) trailing stoploss at 2*ATR
- Discrete position sizing (0.0, ±0.25, ±0.35)
- Volume confirmation filter to reduce false signals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "rsi_hma_mtf_15m_v1"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, adjust=False, min_periods=period//2).mean()
    wma_full = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    
    hma_raw = 2 * wma_half - wma_full
    hma = hma_raw.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    
    return hma.values


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).rolling(window=period, min_periods=period).mean()
    loss_s = pd.Series(loss).rolling(window=period, min_periods=period).mean()
    
    rs = gain_s / (loss_s + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values


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


def calculate_volume_sma(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate volume SMA for confirmation"""
    vol_s = pd.Series(volume).rolling(window=period, min_periods=period).mean()
    return vol_s.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1) ===
    df_4h = get_htf_data(prices, '4h')
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)  # auto shift(1)
    
    # === CALCULATE LTF INDICATORS (vectorized before loop) ===
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    # Calculate 15m HMA for additional trend confirmation
    hma_15m = calculate_hma(close, 21)
    
    # === GENERATE SIGNALS ===
    signals = np.zeros(n)
    
    # Position sizing - discrete levels
    SIZE_ENTRY = 0.30   # 30% for new entries
    SIZE_HALF = 0.15    # 15% for partial exits
    
    # Track position state for stoploss
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 = long, -1 = short, 0 = flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Warmup period
    warmup = max(50, int(np.sqrt(n)))
    
    for i in range(warmup, n):
        # Skip if any indicator is NaN
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_14[i]) or np.isnan(atr_14[i]):
            signals[i] = 0.0
            position_side[i] = position_side[i-1] if i > 0 else 0
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            continue
        
        # === HTF TREND FILTER (4h HMA) ===
        trend_4h = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # === LTF TREND CONFIRMATION (15m HMA) ===
        trend_15m = 1 if close[i] > hma_15m[i] else -1
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > vol_sma_20[i] * 0.8 if not np.isnan(vol_sma_20[i]) else True
        
        # === RSI PULLBACK LOGIC ===
        rsi = rsi_14[i]
        atr = atr_14[i]
        
        # Long entry: 4h uptrend + 15m uptrend + RSI pullback (30-50)
        long_signal = (
            trend_4h == 1 and 
            trend_15m == 1 and 
            30 <= rsi <= 50 and 
            vol_confirmed
        )
        
        # Short entry: 4h downtrend + 15m downtrend + RSI pullback (50-70)
        short_signal = (
            trend_4h == -1 and 
            trend_15m == -1 and 
            50 <= rsi <= 70 and 
            vol_confirmed
        )
        
        # === STOPLOSS LOGIC (Rule 6) ===
        current_side = position_side[i-1] if i > 0 else 0
        current_entry = entry_price[i-1] if i > 0 else 0
        
        stoploss_triggered = False
        
        if current_side == 1 and current_entry > 0:
            # Long position stoploss
            if close[i] < current_entry - 2.0 * atr:
                stoploss_triggered = True
            # Trail stop: update highest since entry
            if i > 0:
                highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
                # Trail at 2*ATR from highest
                if close[i] < highest_since_entry[i] - 2.0 * atr:
                    stoploss_triggered = True
        
        elif current_side == -1 and current_entry > 0:
            # Short position stoploss
            if close[i] > current_entry + 2.0 * atr:
                stoploss_triggered = True
            # Trail stop: update lowest since entry
            if i > 0:
                lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
                # Trail at 2*ATR from lowest
                if close[i] > lowest_since_entry[i] + 2.0 * atr:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
            continue
        
        # === GENERATE SIGNAL ===
        if long_signal and current_side != 1:
            # Enter long
            signals[i] = SIZE_ENTRY
            position_side[i] = 1
            entry_price[i] = close[i]
            highest_since_entry[i] = high[i]
            lowest_since_entry[i] = low[i]
        elif short_signal and current_side != -1:
            # Enter short
            signals[i] = -SIZE_ENTRY
            position_side[i] = -1
            entry_price[i] = close[i]
            highest_since_entry[i] = high[i]
            lowest_since_entry[i] = low[i]
        elif current_side == 1:
            # Hold long position
            signals[i] = SIZE_ENTRY
            position_side[i] = 1
            entry_price[i] = current_entry
            highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else high[i]
            lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else low[i]
        elif current_side == -1:
            # Hold short position
            signals[i] = -SIZE_ENTRY
            position_side[i] = -1
            entry_price[i] = current_entry
            highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else high[i]
            lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else low[i]
        else:
            # Flat
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals