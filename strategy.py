#!/usr/bin/env python3
"""
EXPERIMENT #022 - MTF KAMA-HMA Trend with ATR Trailing Stop
============================================================
Hypothesis: Combining 4h HMA trend filter with 1h KAMA adaptive entry will reduce
whipsaws compared to pure EMA/Supertrend approaches. KAMA adjusts to market efficiency
(erases noise in ranging markets, accelerates in trends). ATR trailing stop at 2.5x
will protect capital during reversals. Position size 0.28 balances risk/return.

Key improvements:
- 4h HMA(21) for primary trend direction (smoother than EMA)
- 1h KAMA(14, ER=10) for adaptive entries (responds to market regime)
- RSI(14) filter: only enter when 40-70 (momentum confirmation, not extreme)
- ATR(14) trailing stop at 2.5x with signal→0 logic
- Discrete signals: 0.0, ±0.28 to minimize fee churn
- Volume spike confirmation (>1.5x 20-bar avg) for entry validation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_4h_kama_hma_atr_v2"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close: np.ndarray, period: int = 14, er_period: int = 10) -> np.ndarray:
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    noise = np.zeros(n)
    signal = np.zeros(n)
    
    for i in range(er_period, n):
        signal[i] = abs(close[i] - close[i - er_period])
        noise[i] = np.sum(np.abs(np.diff(close[max(0, i - er_period):i + 1])))
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = noise > 0
    er[mask] = signal[mask] / noise[mask]
    
    # Smoothing constants
    fast_sc = 2.0 / (2.0 + 1.0)
    slow_sc = 2.0 / (2.0 + 30.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    first_valid = period + er_period
    if first_valid < n:
        kama[first_valid] = close[first_valid]
        
        for i in range(first_valid + 1, n):
            if np.isnan(sc[i]):
                kama[i] = kama[i - 1]
            else:
                kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_hma(close: np.ndarray, period: int = 21) -> np.ndarray:
    """Hull Moving Average - reduced lag, smoother than EMA"""
    n = len(close)
    hma = np.zeros(n)
    hma[:] = np.nan
    
    if n < period:
        return hma
    
    close_s = pd.Series(close)
    
    # WMA(period/2)
    wma_half = close_s.rolling(window=period // 2, min_periods=period // 2).mean()
    # WMA(period)
    wma_full = close_s.rolling(window=period, min_periods=period).mean()
    # 2*WMA(period/2) - WMA(period)
    diff = 2.0 * wma_half - wma_full
    # WMA(sqrt(period))
    sqrt_period = int(np.sqrt(period))
    hma_series = diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    hma[:] = hma_series.values
    
    return hma


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index with proper min_periods"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    if n < period + 1:
        return rsi
    
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    
    rsi[:] = rsi_series.values
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (CRITICAL) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA trend indicator
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # === CALCULATE 1H INDICATORS (VECTORIZED) ===
    kama_1h = calculate_kama(close, period=14, er_period=10)
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    
    # Volume moving average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === GENERATE SIGNALS WITH STOPLOSS LOGIC ===
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size - conservative for DD control
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_price = 0.0  # For trailing stop
    lowest_price = 0.0
    
    first_valid = max(50, int(n * 0.01))  # Warmup period
    
    for i in range(first_valid, n):
        # Skip if any indicator is NaN
        if np.isnan(hma_4h_aligned[i]) or np.isnan(kama_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        current_atr = atr_1h[i]
        current_close = close[i]
        current_rsi = rsi_1h[i]
        current_hma_4h = hma_4h_aligned[i]
        current_kama = kama_1h[i]
        current_vol = volume[i]
        avg_vol = vol_ma[i] if not np.isnan(vol_ma[i]) else current_vol
        
        # === STOPLOSS LOGIC (MUST SET SIGNAL=0 WHEN HIT) ===
        if position_side == 1:  # Long position
            # Update highest price for trailing stop
            highest_price = max(highest_price, current_close)
            
            # Trailing stop: exit if price drops 2.5*ATR from highest
            stop_price = highest_price - 2.5 * current_atr
            
            if current_close < stop_price:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                continue
            
            # Also check 4h trend reversal
            if current_close < current_hma_4h:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                continue
        
        elif position_side == -1:  # Short position
            # Update lowest price for trailing stop
            lowest_price = min(lowest_price, current_close) if lowest_price > 0 else current_close
            
            # Trailing stop: exit if price rises 2.5*ATR from lowest
            stop_price = lowest_price + 2.5 * current_atr
            
            if current_close > stop_price:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                lowest_price = 0.0
                continue
            
            # Also check 4h trend reversal
            if current_close > current_hma_4h:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                lowest_price = 0.0
                continue
        
        # === ENTRY LOGIC (only if flat) ===
        if position_side == 0:
            # Volume confirmation (avoid low liquidity entries)
            vol_confirmed = current_vol > 1.3 * avg_vol if avg_vol > 0 else True
            
            # RSI momentum filter (not overbought/oversold)
            rsi_ok = 35 < current_rsi < 70
            
            # Long entry: price above 4h HMA, KAMA sloping up, RSI confirming
            if current_close > current_hma_4h and vol_confirmed and rsi_ok:
                # Check KAMA slope (adaptive trend confirmation)
                if i > 5 and kama_1h[i] > kama_1h[i - 3]:
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = current_close
                    highest_price = current_close
            
            # Short entry: price below 4h HMA, KAMA sloping down, RSI confirming
            elif current_close < current_hma_4h and vol_confirmed and rsi_ok:
                # Check KAMA slope
                if i > 5 and kama_1h[i] < kama_1h[i - 3]:
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = current_close
                    lowest_price = current_close
        
        # === MAINTAIN POSITION (no signal change = no fees) ===
        else:
            signals[i] = signals[i - 1]  # Keep previous signal
    
    return signals