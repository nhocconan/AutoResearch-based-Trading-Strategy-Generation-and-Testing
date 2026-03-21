#!/usr/bin/env python3
"""
EXPERIMENT #034 - KAMA Trend + RSI Pullback + BB Regime + MACD Momentum + Tighter Stop
====================================================================================
Hypothesis: Replace HMA with KAMA (Kaufman Adaptive Moving Average) which adapts to 
volatility regimes automatically. KAMA flattens during choppy markets and moves fast 
during trends. Combine with 1h RSI entries (less noisy than 15m), BB regime filter 
to avoid squeeze breakouts, and MACD momentum confirmation.

Key changes from #021:
- KAMA(10) instead of HMA - adapts to volatility, fewer whipsaws in chop
- 1h entries instead of 15m - less noise, higher quality signals
- BB Width regime filter - avoid trading during squeeze (low volatility = fake breakouts)
- MACD histogram confirmation - only enter when momentum agrees with trend
- Tighter stoploss: 1.5*ATR instead of 2.0*ATR (KAMA is smoother, can tighten stops)
- Position size: 0.30 instead of 0.35 (more conservative)
- Discrete signal levels: 0.0, ±0.20, ±0.30 to minimize churn costs

Why this might beat Sharpe=11.523:
- KAMA's adaptive nature reduces false signals in choppy markets
- 1h timeframe has fewer whipsaws than 15m while still catching moves
- BB regime filter avoids low-volatility traps
- MACD confirmation adds momentum filter
- Tighter stops protect capital better in reversals
"""

import numpy as np
import pandas as pd

name = "mtf_kama_rsi_bb_macd_atr_tp_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - moves fast in trends, flat in chop
    """
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(slow_period, n):
        signal = abs(close[i] - close[i - slow_period])
        noise = np.sum(np.abs(np.diff(close[i - slow_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(slow_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[slow_period] = close[slow_period]
    for i in range(slow_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period] = np.mean(tr[1:period + 1])
    
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
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
    # Handle division by zero
    rsi[avg_loss == 0] = 100
    
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    macd_signal = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    macd_hist = macd_line - macd_signal
    
    # Fill NaN with 0
    macd_line = np.nan_to_num(macd_line, 0)
    macd_signal = np.nan_to_num(macd_signal, 0)
    macd_hist = np.nan_to_num(macd_hist, 0)
    
    return macd_line, macd_signal, macd_hist


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / middle
    
    # Fill NaN
    middle = np.nan_to_num(middle, 0)
    upper = np.nan_to_num(upper, 0)
    lower = np.nan_to_num(lower, 0)
    bandwidth = np.nan_to_num(bandwidth, 0)
    
    return upper, middle, lower, bandwidth


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    if n < 100:
        return np.zeros(n)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    kama_10_1h = calculate_kama(close, period=10)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper, bb_middle, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, period=20)
    
    # 4h KAMA for trend filter (resample 1h → 4h)
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
    n_4h = len(c_4h)
    
    if n_4h < 50:
        return np.zeros(n)
    
    # Calculate 4h KAMA for trend
    kama_10_4h = calculate_kama(c_4h, period=10)
    kama_30_4h = calculate_kama(c_4h, period=30)
    
    # 4h trend direction based on KAMA cross
    trend_4h = np.zeros(n_4h)
    for i in range(30, n_4h):
        if kama_10_4h[i] > kama_30_4h[i] and c_4h[i] > kama_10_4h[i]:
            trend_4h[i] = 1  # Bullish
        elif kama_10_4h[i] < kama_30_4h[i] and c_4h[i] < kama_10_4h[i]:
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
    SIZE_FULL = 0.30   # Full position (reduced from 0.35)
    SIZE_HALF = 0.15   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # BB bandwidth thresholds for regime filter
    BB_MIN_BW = 0.02      # Minimum bandwidth to trade (avoid squeeze)
    BB_MAX_BW = 0.15      # Maximum bandwidth (avoid extreme volatility)
    
    # MACD histogram threshold for momentum confirmation
    MACD_MIN_HIST = 0     # Histogram must be positive for long, negative for short
    
    # ATR stoploss multiplier (TIGHTER than #021)
    ATR_STOP_MULT = 1.5   # Tighter stoploss (KAMA is smoother)
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.02  # Target 2% ATR
    
    first_valid = max(50, 30, 14, 20, 26)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    trailing_stop = np.zeros(n)  # Track trailing stop level
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bb_bandwidth[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        bb_w = bb_bandwidth[i]
        macd_h = macd_hist[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Avoid extremely high/low volatility regimes
        if bb_w < BB_MIN_BW or bb_w > BB_MAX_BW:
            if i > 0 and position_side[i - 1] != 0:
                # Check stoploss even in bad regime
                prev_side = position_side[i - 1]
                prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
                
                if prev_side == 1:
                    stoploss_price = trailing_stop[i-1] if trailing_stop[i-1] > 0 else prev_entry - ATR_STOP_MULT * atr
                    if price < stoploss_price:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        trailing_stop[i] = 0
                        continue
                elif prev_side == -1:
                    stoploss_price = trailing_stop[i-1] if trailing_stop[i-1] > 0 else prev_entry + ATR_STOP_MULT * atr
                    if price > stoploss_price:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        trailing_stop[i] = 0
                        continue
                
                # Hold position through bad regime
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                trailing_stop[i] = trailing_stop[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_trail = trailing_stop[i - 1] if trailing_stop[i - 1] > 0 else prev_entry
            
            if prev_side == 1:
                # Update trailing stop (move up only)
                new_trail = max(prev_trail, prev_entry + ATR_STOP_MULT * atr)
                trailing_stop[i] = new_trail
                
                # Stoploss check
                if price < new_trail:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_stop[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    trailing_stop[i] = prev_entry  # Trail to breakeven
                    continue
                
                # RSI exit signal
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_stop[i] = 0
                    continue
                    
            elif prev_side == -1:
                # Update trailing stop (move down only)
                new_trail = min(prev_trail, prev_entry - ATR_STOP_MULT * atr) if prev_trail > 0 else prev_entry - ATR_STOP_MULT * atr
                trailing_stop[i] = new_trail
                
                # Stoploss check
                if price > new_trail:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_stop[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    trailing_stop[i] = prev_entry  # Trail to breakeven
                    continue
                
                # RSI exit signal
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_stop[i] = 0
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
            # RSI pullback entry + MACD confirmation
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30 and macd_h > MACD_MIN_HIST:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                trailing_stop[i] = price - ATR_STOP_MULT * atr
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    trailing_stop[i] = trailing_stop[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_stop[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # RSI rally entry + MACD confirmation
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70 and macd_h < -MACD_MIN_HIST:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                trailing_stop[i] = price + ATR_STOP_MULT * atr
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    trailing_stop[i] = trailing_stop[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_stop[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            trailing_stop[i] = 0
    
    return signals