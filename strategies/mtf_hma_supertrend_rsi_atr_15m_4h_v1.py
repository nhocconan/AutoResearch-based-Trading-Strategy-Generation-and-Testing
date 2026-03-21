#!/usr/bin/env python3
"""
EXPERIMENT #025 - MTF HMA+Supertrend+RSI+ATR Dynamic Sizing (15m+4h Simplified v1)
==================================================================================================
Hypothesis: Simplify the complex #040 strategy while maintaining edge.
- Remove ADX, KAMA, BBW, Z-score filters (too many conditions reduce trade frequency)
- Keep proven 15m entries + 4h trend filter (worked in #031, #034, #035)
- Use mtf_data helper for proper MTF alignment (CRITICAL - many strategies failed audit)
- ATR-based dynamic position sizing (smaller size when volatility is high)
- Tighter stoploss (1.5*ATR vs 2.0*ATR) for better R:R
- RSI range: 35-65 (wider than #040's 40-60 for more entries)

Why this should work:
- Fewer filters = more trades while maintaining quality
- 4h trend is more stable than 1h (less whipsaw)
- ATR dynamic sizing reduces risk in high volatility periods
- Simpler logic = fewer bugs and edge cases
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_supertrend_rsi_atr_15m_4h_v1"
timeframe = "15m"
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
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).rolling(window=half, min_periods=half).apply(
        lambda x: np.sum(x * np.arange(1, half + 1)) / np.sum(np.arange(1, half + 1)), raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1)), raw=True
    ).values
    
    raw_hma = 2 * wma1 - wma2
    
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, sqrt_period + 1)) / np.sum(np.arange(1, sqrt_period + 1)), raw=True
    ).values
    
    return np.nan_to_num(hma)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    return np.nan_to_num(rsi)


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    upper = (high + low) / 2 + multiplier * atr
    lower = (high + low) / 2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)
    
    supertrend[0] = lower[0]
    
    for i in range(1, n):
        if direction[i - 1] == 1:
            supertrend[i] = max(lower[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper[i]
                direction[i] = -1
        else:
            supertrend[i] = min(upper[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower[i]
                direction[i] = 1
    
    return supertrend, direction


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get 4h data for trend filter using mtf_data helper
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 15m indicators (entry timing)
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_dir_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # 4h indicators (trend filter) - using mtf_data helper
    hma_4h = calculate_hma(close_4h, period=21)
    supertrend_4h, st_dir_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    
    # Align 4h indicators to 15m timeframe
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    st_dir_4h_aligned = align_htf_to_ltf(prices, df_4h, st_dir_4h)
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing parameters
    BASE_SIZE = 0.35
    TARGET_ATR_PCT = 0.02  # Target ATR as % of price
    
    # Entry thresholds
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # Stoploss
    ATR_STOP_MULT = 1.5
    
    first_valid = max(100, 40 * 16)  # Need enough 4h bars
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for valid data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get 4h trend
        trend_4h = 0
        if close_4h_aligned[i] > hma_4h_aligned[i]:
            trend_4h = 1
        elif close_4h_aligned[i] < hma_4h_aligned[i]:
            trend_4h = -1
        
        st_trend_4h = st_dir_4h_aligned[i]
        
        # ATR-based dynamic position sizing
        atr_pct = atr_15m[i] / close[i] if close[i] > 0 else 0
        if atr_pct > 0:
            size_multiplier = min(1.0, TARGET_ATR_PCT / atr_pct)
        else:
            size_multiplier = 1.0
        
        current_size = BASE_SIZE * size_multiplier
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, close[i])
                current_low = min(prev_low, close[i]) if prev_low > 0 else close[i]
            else:
                current_high = max(prev_high, close[i]) if prev_high > 0 else close[i]
                current_low = min(prev_low, close[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (1.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] >= tp_price:
                    signals[i] = current_size / 2
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] <= tp_price:
                    signals[i] = -current_size / 2
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
                    if close[i] > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 15m RSI pullback
        if trend_4h == 1 and st_trend_4h == 1:  # Bullish trend on 4h
            if RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX:  # Pullback entry
                signals[i] = current_size
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
                
        elif trend_4h == -1 and st_trend_4h == -1:  # Bearish trend on 4h
            if RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX:  # Pullback entry
                signals[i] = -current_size
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals