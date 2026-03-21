#!/usr/bin/env python3
"""
EXPERIMENT #032 - HMA Trend + RSI Pullback + Bollinger Regime Filter (1h timeframe)
====================================================================================
Hypothesis: Building on #021's success, switch from 15m to 1h timeframe to reduce noise
and transaction costs. Add Bollinger Band width regime filter to avoid trading during
low-volatility consolidation periods. Adjust position sizing to 0.30 (more conservative).

Key changes from #021:
- 1h timeframe instead of 15m - fewer signals, less fee drag
- 4h trend + 1h entries (proven MTF combination)
- Bollinger Band width filter - avoid trading when BW < 20th percentile
- RSI thresholds: 40/60 instead of 45/55 (wait for deeper pullbacks)
- ATR stop multiplier: 2.5 instead of 2.0 (looser stops, fewer premature exits)
- Position size: 0.30 instead of 0.35 (more conservative)
- Simplified take-profit: reduce to half at 2R, trail stop to 1R

Why this might beat Sharpe=11.523:
- 1h has less noise than 15m, fewer false signals
- BB width filter avoids choppy markets where trend strategies fail
- Looser stops reduce whipsaw exits
- Lower position size reduces drawdown risk
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_bb_regime_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average - reduces lag vs EMA"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    if half < 1:
        half = 1
    
    # WMA helper function
    def wma(arr, w):
        result = np.zeros(len(arr))
        weights = np.arange(1, w + 1, dtype=float)
        w_sum = np.sum(weights)
        for i in range(w - 1, len(arr)):
            result[i] = np.sum(arr[i - w + 1:i + 1] * weights) / w_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
    return hma


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
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


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = (avg_loss > 0) & (avg_gain >= 0)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    # Handle edge case where avg_loss = 0
    rsi[avg_loss == 0] = 100
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    bw = (upper - lower) / mean  # Band width as % of price
    
    return upper, lower, bw


def calculate_bb_percentile(bw, lookback=100):
    """Calculate Bollinger Band width percentile for regime filter"""
    n = len(bw)
    bb_pct = np.zeros(n)
    
    for i in range(lookback, n):
        window = bw[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            bb_pct[i] = np.sum(valid <= bw[i]) / len(valid)
    
    return bb_pct


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_pct = calculate_bb_percentile(bb_width, lookback=100)
    
    hma_16_1h = calculate_hma(close, period=16)
    hma_48_1h = calculate_hma(close, period=48)
    
    # 4h HMA for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # Calculate 4h HMA for trend
    hma_16_4h = calculate_hma(c_4h, period=16)
    hma_48_4h = calculate_hma(c_4h, period=48)
    
    # 4h trend direction based on HMA cross + price position
    trend_4h = np.zeros(len(c_4h))
    for i in range(48, len(c_4h)):
        if hma_16_4h[i] > hma_48_4h[i] and c_4h[i] > hma_16_4h[i]:
            trend_4h[i] = 1  # Bullish
        elif hma_16_4h[i] < hma_48_4h[i] and c_4h[i] < hma_16_4h[i]:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.30   # Full position (more conservative than 0.35)
    SIZE_HALF = 0.15   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 40   # Enter long on deeper pullback in uptrend
    RSI_SHORT_ENTRY = 60  # Enter short on stronger rally in downtrend
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # Bollinger Band width regime filter
    BB_PCT_MIN = 0.20     # Only trade when BB width > 20th percentile
    
    # ATR stoploss multiplier (looser than #021)
    ATR_STOP_MULT = 2.5   # 2.5*ATR stop
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.015  # Target 1.5% ATR
    
    first_valid = max(100, 48, 14, 20)  # Wait for all indicators
    
    # Track state for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    stop_price = np.zeros(n)    # Current stop price
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bb_pct[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        bb_pct_val = bb_pct[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Bollinger Band width regime filter - avoid low volatility
        if bb_pct_val < BB_PCT_MIN:
            # Close existing positions if in low vol regime
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                stop_price[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                stop_price[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_stop = stop_price[i - 1] if stop_price[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                
                # Calculate trailing stop (trail at 1R after TP hit)
                if prev_tp:
                    # Trail stop at 1R profit (breakeven + 1R)
                    trail_stop = prev_entry + ATR_STOP_MULT * atr
                    current_stop = max(prev_stop, trail_stop)
                else:
                    current_stop = prev_entry - ATR_STOP_MULT * atr
                
                stop_price[i] = current_stop
                
                # Stoploss check
                if price < current_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    stop_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    # Reduce to half position
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    stop_price[i] = prev_entry  # Trail to breakeven
                    highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else price
                    lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                    continue
                
                # RSI exit signal
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    stop_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                
                # Calculate trailing stop (trail at 1R after TP hit)
                if prev_tp:
                    # Trail stop at 1R profit (breakeven - 1R)
                    trail_stop = prev_entry - ATR_STOP_MULT * atr
                    current_stop = min(prev_stop, trail_stop) if prev_stop > 0 else trail_stop
                else:
                    current_stop = prev_entry + ATR_STOP_MULT * atr
                
                stop_price[i] = current_stop
                
                # Stoploss check
                if price > current_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    stop_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    # Reduce to half position
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    stop_price[i] = prev_entry  # Trail to breakeven
                    highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                    lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                    continue
                
                # RSI exit signal
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    stop_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            
            # Hold position if no exit trigger
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            stop_price[i] = stop_price[i - 1]
            highest_since_entry[i] = highest_since_entry[i-1]
            lowest_since_entry[i] = lowest_since_entry[i-1]
            continue
        
        # Dynamic position sizing based on ATR volatility
        current_atr_pct = atr / price if price > 0 else 0
        if current_atr_pct > 0:
            size_multiplier = min(1.5, max(0.5, TARGET_ATR_PCT / current_atr_pct))
        else:
            size_multiplier = 1.0
        
        position_size = SIZE_FULL * size_multiplier
        position_size = min(SIZE_FULL, max(SIZE_HALF, position_size))  # Clamp
        
        if trend == 1:  # 4h uptrend
            # RSI pullback entry in uptrend
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                stop_price[i] = price - ATR_STOP_MULT * atr
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                stop_price[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # RSI rally entry in downtrend
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                stop_price[i] = price + ATR_STOP_MULT * atr
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                stop_price[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            stop_price[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals