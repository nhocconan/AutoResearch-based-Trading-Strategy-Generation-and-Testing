#!/usr/bin/env python3
"""
EXPERIMENT #022 - Supertrend + MACD-RSI Dual Confirmation + BBW Regime Filter
====================================================================================
Hypothesis: Replace HMA trend with Supertrend for more definitive trend signals.
Supertrend provides clear stop levels and reduces whipsaws in choppy markets.
Add MACD histogram confirmation alongside RSI for dual entry filter.
Use Bollinger Band Width percentile to detect volatility regime and avoid extreme conditions.

Key improvements over #021:
- Supertrend(ATR=10, mult=3) instead of HMA cross - clearer trend definition
- MACD histogram + RSI dual confirmation - fewer false entries
- BBW percentile filter - avoid trading during volatility extremes
- Trailing stop that tightens after 1R profit
- Discrete signal levels: 0.0, ±0.25, ±0.35 to minimize churn costs

Why this might beat Sharpe=5.4:
- Supertrend provides built-in stop levels, reducing drawdown
- Dual confirmation (MACD+RSI) filters out weak signals
- BBW regime detection avoids trading during extreme volatility
- Better risk-reward with trailing stop management
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_macd_rsi_bbw_v1"
timeframe = "1h"
leverage = 1.0


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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator with trend direction"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1 if close[i] < supertrend[i] else 1
        else:
            # Update supertrend based on previous trend
            if trend[i - 1] == 1:
                supertrend[i] = max(upper_band[i], supertrend[i - 1]) if upper_band[i] < supertrend[i - 1] else upper_band[i]
                supertrend[i] = lower_band[i] if close[i] < supertrend[i - 1] else supertrend[i]
            else:
                supertrend[i] = min(lower_band[i], supertrend[i - 1]) if lower_band[i] > supertrend[i - 1] else lower_band[i]
                supertrend[i] = upper_band[i] if close[i] > supertrend[i - 1] else supertrend[i]
            
            # Determine current trend
            trend[i] = 1 if close[i] > supertrend[i] else -1
    
    return supertrend, trend, upper_band, lower_band


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma if np.any(sma > 0) else np.zeros(n)
    
    # Calculate bandwidth percentile over rolling window
    bw_percentile = np.zeros(n)
    lookback = 100
    for i in range(lookback, n):
        bw_window = bandwidth[i - lookback:i + 1]
        bw_percentile[i] = np.sum(bw_window <= bandwidth[i]) / len(bw_window)
    
    return upper, lower, bandwidth, bw_percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper, bb_lower, bb_width, bb_percentile = calculate_bollinger_bands(close, period=20)
    supertrend, st_trend, st_upper, st_lower = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # 4h Supertrend for trend filter (resample 1h → 4h)
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
    
    # Calculate 4h Supertrend for trend
    _, st_trend_4h, _, _ = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(st_trend_4h):
            trend_1h[i] = st_trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.20   # Half position (after take profit)
    SIZE_QUARTER = 0.10  # Quarter position (trailing)
    
    # Entry thresholds
    RSI_LONG_ENTRY = 40   # Enter long on pullback
    RSI_SHORT_ENTRY = 60  # Enter short on rally
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # MACD histogram confirmation
    MACD_MIN_HIST = 0     # Histogram must be positive for long, negative for short
    
    # BBW regime filter
    BBW_MIN_PERCENTILE = 0.15  # Don't trade in bottom 15% (too quiet)
    BBW_MAX_PERCENTILE = 0.85  # Don't trade in top 15% (too volatile)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    ATR_TRAIL_MULT = 1.5  # Trailing stop after 1R profit
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    first_valid = max(100, 48, 14, 26)  # Wait for all indicators
    
    # Track position state
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    trailing_active = np.zeros(n)  # Track if trailing stop is active
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        macd_h = macd_hist[i]
        bb_pct = bb_percentile[i]
        atr = atr_1h[i]
        price = close[i]
        st_val = supertrend[i]
        
        # BBW regime filter - avoid extreme volatility conditions
        if bb_pct < BBW_MIN_PERCENTILE or bb_pct > BBW_MAX_PERCENTILE:
            if i > 0 and position_side[i - 1] != 0:
                # Check stoploss even in filtered regime
                prev_side = position_side[i - 1]
                prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
                
                if prev_side == 1:
                    stoploss_price = prev_entry - ATR_STOP_MULT * atr
                    if price < stoploss_price:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        trailing_active[i] = 0
                        continue
                elif prev_side == -1:
                    stoploss_price = prev_entry + ATR_STOP_MULT * atr
                    if price > stoploss_price:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        trailing_active[i] = 0
                        continue
                
                # Hold position through filter
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                trailing_active[i] = trailing_active[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Check existing positions for stoploss/takeprofit/trailing
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_trail = trailing_active[i - 1]
            
            if prev_side == 1:  # Long position
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                
                # Calculate profit in R multiples
                profit_r = (price - prev_entry) / (ATR_STOP_MULT * atr) if atr > 0 else 0
                
                # Stoploss check (initial or trailing)
                if prev_trail:
                    # Trailing stop at 1R below highest
                    trail_stop = highest_since_entry[i] - ATR_TRAIL_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        trailing_active[i] = 0
                        highest_since_entry[i] = 0
                        continue
                else:
                    # Initial stoploss
                    stoploss_price = prev_entry - ATR_STOP_MULT * atr
                    if price < stoploss_price:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        trailing_active[i] = 0
                        highest_since_entry[i] = 0
                        continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    trailing_active[i] = 1  # Activate trailing after TP
                    highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else price
                    continue
                
                # Activate trailing stop after 1R profit
                if not prev_trail and profit_r >= 1.0:
                    trailing_active[i] = 1
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else price
                    continue
                
                # RSI exit signal (overbought)
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_active[i] = 0
                    highest_since_entry[i] = 0
                    continue
                
                # Supertrend reversal exit
                if trend == -1 and prev_side == 1:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_active[i] = 0
                    highest_since_entry[i] = 0
                    continue
                
                # Hold position
                signals[i] = signals[i - 1]
                position_side[i] = 1
                entry_price[i] = prev_entry
                tp_triggered[i] = prev_tp
                trailing_active[i] = prev_trail
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else price
                continue
                
            elif prev_side == -1:  # Short position
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                
                # Calculate profit in R multiples
                profit_r = (prev_entry - price) / (ATR_STOP_MULT * atr) if atr > 0 else 0
                
                # Stoploss check (initial or trailing)
                if prev_trail:
                    # Trailing stop at 1R above lowest
                    trail_stop = lowest_since_entry[i] + ATR_TRAIL_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        trailing_active[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                else:
                    # Initial stoploss
                    stoploss_price = prev_entry + ATR_STOP_MULT * atr
                    if price > stoploss_price:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        trailing_active[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    trailing_active[i] = 1  # Activate trailing after TP
                    lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                    continue
                
                # Activate trailing stop after 1R profit
                if not prev_trail and profit_r >= 1.0:
                    trailing_active[i] = 1
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                    continue
                
                # RSI exit signal (oversold)
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_active[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Supertrend reversal exit
                if trend == 1 and prev_side == -1:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_active[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Hold position
                signals[i] = signals[i - 1]
                position_side[i] = -1
                entry_price[i] = prev_entry
                tp_triggered[i] = prev_tp
                trailing_active[i] = prev_trail
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
                continue
        
        # New entry logic - dual confirmation required
        if trend == 1:  # 4h uptrend - look for long entries
            # RSI pullback + MACD positive histogram
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 25 and macd_h > MACD_MIN_HIST:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                trailing_active[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                trailing_active[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                    
        elif trend == -1:  # 4h downtrend - look for short entries
            # RSI rally + MACD negative histogram
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 75 and macd_h < -MACD_MIN_HIST:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                trailing_active[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                trailing_active[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            trailing_active[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals