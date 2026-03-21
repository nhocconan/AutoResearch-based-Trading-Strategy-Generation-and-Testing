#!/usr/bin/env python3
"""
EXPERIMENT #018 - MTF EMA+MACD+Volume+RSI+ATR (1h+4h Clean v1)
==================================================================================================
Hypothesis: Current best (#012) uses 1h+4h with Sharpe=0.478. #040 uses 15m+1h but may have bugs.
This experiment simplifies the MTF logic while keeping proven components:
- 4h EMA(21/55) crossover for trend direction (cleaner than HMA/Supertrend combo)
- 1h MACD histogram cross for entry timing (momentum confirmation)
- Volume spike filter (volume > 1.5x 20-bar MA) for conviction
- 1h RSI(14) 40-60 range for pullback entries (avoid chasing)
- ATR(14) 2.5x stoploss (wider than #040's 2.0x to reduce premature exits)
- Position size: 0.30 (conservative, proven safe)

Why this should beat #012 (Sharpe=0.478):
- EMA crossover is more stable than HMA for trend
- MACD histogram cross is proven momentum signal
- Volume filter adds conviction (missing in #012)
- Simpler logic = fewer bugs than #040's complex state tracking
- 1h timeframe is more stable than 15m for live trading
"""

import numpy as np
import pandas as pd

name = "mtf_ema_macd_volume_rsi_atr_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = np.zeros(n)
    multiplier = 2.0 / (period + 1)
    
    # Initialize with SMA
    ema[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD (line, signal, histogram)"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line is EMA of MACD line
    signal_line = np.zeros(n)
    first_valid = slow + signal - 1
    
    # Initialize signal with first valid MACD values
    valid_macd = macd_line[slow-1:first_valid+1]
    if len(valid_macd) > 0:
        signal_line[first_valid] = np.mean(valid_macd)
    
    multiplier = 2.0 / (signal + 1)
    for i in range(first_valid + 1, n):
        if macd_line[i] != 0:
            signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]
        else:
            signal_line[i] = signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_volume_ma(volume, period=20):
    """Calculate Volume Moving Average"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_ma = np.zeros(n)
    for i in range(period - 1, n):
        vol_ma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_ma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    macd_line_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    vol_ma_1h = calculate_volume_ma(volume, period=20)
    
    # Resample to 4h for trend filters
    prices_indexed = prices.set_index('open_time')
    
    # Resample to 4h using proper aggregation
    df_4h = prices_indexed.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    if len(df_4h) < 100:
        return np.zeros(n)
    
    # Calculate 4h trend indicators
    close_4h = df_4h['close'].values
    ema21_4h = calculate_ema(close_4h, 21)
    ema55_4h = calculate_ema(close_4h, 55)
    
    # Determine 4h trend direction
    trend_4h = np.zeros(len(close_4h))
    for i in range(55, len(close_4h)):
        if close_4h[i] > ema21_4h[i] and ema21_4h[i] > ema55_4h[i]:
            trend_4h[i] = 1  # Bullish
        elif close_4h[i] < ema21_4h[i] and ema21_4h[i] < ema55_4h[i]:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h using reindex with ffill
    trend_4h_series = pd.Series(trend_4h, index=df_4h.index)
    trend_1h_aligned = trend_4h_series.reindex(prices_indexed.index, method='ffill').values
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Entry thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    VOLUME_MULT = 1.5
    ATR_STOP_MULT = 2.5
    ATR_TP_MULT = 2.0  # Take profit at 2R
    
    # Warmup period
    first_valid = max(200, 55 * 4)  # Need 4h EMA55 to be valid
    
    # Generate signals
    signals = np.zeros(n)
    
    # Track position state for stoploss/TP
    position_side = 0
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_1h_aligned[i]
        rsi_val = rsi_1h[i]
        macd_hist = macd_hist_1h[i]
        macd_hist_prev = macd_hist_1h[i - 1] if i > 0 else 0
        vol_ratio = volume[i] / vol_ma_1h[i] if vol_ma_1h[i] > 0 else 0
        price = close[i]
        atr = atr_1h[i]
        
        # Check existing position for stoploss/TP
        if position_side != 0:
            # Update highest/lowest since entry
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, price)
                lowest_since_entry = min(lowest_since_entry, price) if lowest_since_entry > 0 else price
            else:
                highest_since_entry = max(highest_since_entry, price) if highest_since_entry > 0 else price
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Stoploss check
            if position_side == 1:
                stoploss_price = entry_price - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit at 2R - reduce to half
                tp_price = entry_price + ATR_TP_MULT * atr
                if not tp_triggered and price >= tp_price:
                    signals[i] = SIZE_HALF
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R after TP
                if tp_triggered:
                    trail_stop = highest_since_entry - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
            
            elif position_side == -1:
                stoploss_price = entry_price + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit at 2R - reduce to half
                tp_price = entry_price - ATR_TP_MULT * atr
                if not tp_triggered and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R after TP
                if tp_triggered:
                    trail_stop = lowest_since_entry + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
            
            # Hold position if no exit
            signals[i] = signals[i - 1] if i > 0 else 0.0
            continue
        
        # Entry logic: 4h trend + 1h MACD cross + Volume + RSI pullback
        if trend == 1:  # Bullish 4h trend
            # MACD histogram cross above zero (momentum turning positive)
            macd_cross_long = macd_hist_prev <= 0 and macd_hist > 0
            # Or MACD already positive and rising
            macd_momentum_long = macd_hist > 0 and macd_hist > macd_hist_prev
            
            if (macd_cross_long or macd_momentum_long):
                # Volume confirmation
                if vol_ratio >= VOLUME_MULT:
                    # RSI pullback (not overbought)
                    if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                        signals[i] = SIZE_FULL
                        position_side = 1
                        entry_price = price
                        tp_triggered = False
                        highest_since_entry = price
                        lowest_since_entry = price
                        continue
        
        elif trend == -1:  # Bearish 4h trend
            # MACD histogram cross below zero (momentum turning negative)
            macd_cross_short = macd_hist_prev >= 0 and macd_hist < 0
            # Or MACD already negative and falling
            macd_momentum_short = macd_hist < 0 and macd_hist < macd_hist_prev
            
            if (macd_cross_short or macd_momentum_short):
                # Volume confirmation
                if vol_ratio >= VOLUME_MULT:
                    # RSI pullback (not oversold)
                    if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                        signals[i] = -SIZE_FULL
                        position_side = -1
                        entry_price = price
                        tp_triggered = False
                        highest_since_entry = price
                        lowest_since_entry = price
                        continue
        
        # No position
        signals[i] = 0.0
    
    return signals