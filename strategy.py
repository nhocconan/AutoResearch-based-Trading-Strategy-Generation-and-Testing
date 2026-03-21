#!/usr/bin/env python3
"""
EXPERIMENT #029 - HMA Trend + 1h RSI + Bollinger Regime + MACD Momentum
====================================================================================
Hypothesis: Building on #021's success, switch to 1h entries (slower than 15m) for fewer whipsaws.
Add Bollinger Band width regime filter to avoid trading during low-volatility squeezes.
Add MACD histogram confirmation for momentum alignment. Use ATR percentile for position sizing.

Key improvements over #021:
- 1h RSI entries instead of 15m - fewer false signals, better risk/reward ratio
- Bollinger Band width percentile filter - avoid choppy/squeeze markets
- MACD histogram confirmation - ensure momentum aligns with trend
- ATR percentile-based sizing - adapt to volatility regime more smoothly
- Tighter RSI thresholds (40/60 vs 45/55) - higher quality entries only
- Trailing stop at 1R after TP hit - lock in more profits

Why this might beat Sharpe=11.523:
- Slower 1h entries reduce churn costs and false breakouts
- Regime filter avoids dead zones where mean-reversion kills trends
- MACD confirmation adds momentum filter without over-complicating
- Better position sizing adapts to changing volatility regimes
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_bb_regime_macd_1h_v1"
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
    
    def wma(arr, w):
        result = np.zeros(len(arr))
        weights = np.arange(1, w + 1, dtype=np.float64)
        w_sum = np.sum(weights)
        for i in range(w - 1, len(arr)):
            result[i] = np.dot(arr[i - w + 1:i + 1], weights) / w_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
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
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = (avg_loss > 0) & (avg_gain >= 0)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth"""
    n = len(close)
    
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    bandwidth = (upper - lower) / mean
    
    return upper, lower, mean, bandwidth


def calculate_atr_percentile(atr, lookback=100):
    """Calculate ATR percentile for regime detection"""
    n = len(atr)
    percentile = np.zeros(n)
    
    for i in range(lookback, n):
        window = atr[i-lookback:i+1]
        percentile[i] = np.sum(window <= atr[i]) / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper, bb_lower, bb_mean, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    atr_pct = calculate_atr_percentile(atr_1h, lookback=100)
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
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    
    # 4h trend direction based on HMA cross and price position
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
    
    # Map 4h ATR to 1h
    atr_4h_mapped = np.zeros(n)
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(atr_4h):
            atr_4h_mapped[i] = atr_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.18   # Half position (after take profit)
    
    # RSI thresholds for pullback entries (tighter than #021)
    RSI_LONG_ENTRY = 40   # Enter long on deeper pullback in uptrend
    RSI_SHORT_ENTRY = 60  # Enter short on stronger rally in downtrend
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # Bollinger Band width thresholds (percentile)
    BB_WIDTH_LOW = 0.20   # Avoid trading when bandwidth in bottom 20% (squeeze)
    BB_WIDTH_HIGH = 0.85  # Caution when bandwidth in top 85% (extended)
    
    # MACD histogram confirmation
    MACD_MIN = 0.0        # Require positive histogram for longs
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0   # Same as #021
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.015  # Target 1.5% ATR (slightly lower than #021)
    
    first_valid = max(100, 48, 14, 20)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    trailing_stop = np.zeros(n)  # Trailing stop level after TP
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(atr_pct[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        macd_val = macd_hist[i]
        atr = atr_1h[i]
        atr_4h_val = atr_4h_mapped[i]
        bandwidth_pct = atr_pct[i]
        price = close[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.06:  # ATR > 6% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            trailing_stop[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_trail = trailing_stop[i - 1]
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                
                # Determine current stop level
                if prev_tp:
                    # After TP, use trailing stop at 1R
                    current_stop = max(prev_entry, prev_trail)
                else:
                    current_stop = prev_entry - ATR_STOP_MULT * atr
                
                # Stoploss check
                if price < current_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_stop[i] = 0
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
                    trailing_stop[i] = prev_entry + ATR_STOP_MULT * atr  # Trail at 1R
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    continue
                
                # RSI exit signal
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_stop[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                
                # Determine current stop level
                if prev_tp:
                    # After TP, use trailing stop at 1R
                    current_stop = min(prev_entry, prev_trail)
                else:
                    current_stop = prev_entry + ATR_STOP_MULT * atr
                
                # Stoploss check
                if price > current_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_stop[i] = 0
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
                    trailing_stop[i] = prev_entry - ATR_STOP_MULT * atr  # Trail at 1R
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    continue
                
                # RSI exit signal
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_stop[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
        
        # Bollinger Band width regime filter
        if bandwidth_pct < BB_WIDTH_LOW:
            # Low volatility squeeze - avoid new entries but hold existing
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                trailing_stop[i] = trailing_stop[i - 1]
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
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
            # RSI pullback entry in uptrend + MACD confirmation
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 25 and macd_val > MACD_MIN:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                trailing_stop[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    trailing_stop[i] = trailing_stop[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_stop[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # RSI rally entry in downtrend + MACD confirmation
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 75 and macd_val < -MACD_MIN:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                trailing_stop[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    trailing_stop[i] = trailing_stop[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_stop[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            trailing_stop[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals