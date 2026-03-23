# Strategy: mtf_supertrend_macd_zscore_4h_1d_v1

## Status
ACTIVE - Sharpe=0.190 | Return=+50.8% | DD=-24.1%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.196 | +10.1% | -17.1% | 29 |
| ETHUSDT | -0.066 | +13.8% | -18.1% | 29 |
| SOLUSDT | 0.833 | +128.6% | -37.1% | 12 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.153 | +4.2% | -8.0% | 13 |
| ETHUSDT | -0.530 | -2.7% | -12.9% | 14 |
| SOLUSDT | -0.381 | -1.6% | -23.5% | 14 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #010 - Supertrend+MACD+Zscore 4h+1d v1
==================================================================================================
Hypothesis: Use 4h primary timeframe (fewer trades, less fee drag) with 1d SMA(50) trend filter
+ 4h Supertrend direction + MACD momentum confirmation + Z-score filter. 4h should capture
swing moves with cleaner signals than 1h/30m. Daily filter avoids trading against major trend.

Why this should work:
- 4h timeframe: fewer false signals, less fee drag, better risk/reward
- Supertrend: proven trend follower with ATR-based stops built into logic
- MACD histogram: momentum confirmation to avoid entering at trend exhaustion
- Z-score(20): filter out extreme overbought/oversold conditions
- Daily SMA(50): major trend filter - only trade in direction of daily trend
- Discrete signal levels (0.0, ±0.25, ±0.35) minimize churn costs
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_macd_zscore_4h_1d_v1"
timeframe = "4h"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend_values, trend_direction (1=bullish, -1=bearish)
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2.0
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    # Initialize
    upper_band[period - 1] = hl2[period - 1] + multiplier * atr[period - 1]
    lower_band[period - 1] = hl2[period - 1] - multiplier * atr[period - 1]
    supertrend[period - 1] = upper_band[period - 1]
    trend[period - 1] = -1  # Start bearish
    
    for i in range(period, n):
        # Calculate new bands
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        # Update bands based on previous trend
        if trend[i - 1] == 1:
            upper_band[i] = min(upper_band[i], upper_band[i - 1])
        else:
            lower_band[i] = max(lower_band[i], lower_band[i - 1])
        
        # Determine trend
        if close[i] > upper_band[i - 1]:
            trend[i] = 1
            supertrend[i] = lower_band[i]
        elif close[i] < lower_band[i - 1]:
            trend[i] = -1
            supertrend[i] = upper_band[i]
        else:
            trend[i] = trend[i - 1]
            supertrend[i] = supertrend[i - 1]
    
    return supertrend, trend


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # Calculate EMAs
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(period - 1, n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
        else:
            zscore[i] = 0.0
    
    return zscore


def calculate_sma(close, period=50):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 4h indicators for entry timing
    atr_4h = calculate_atr(high, low, close, period=14)
    supertrend_4h, st_trend_4h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    zscore_4h = calculate_zscore(close, period=20)
    
    # Get 1d data using mtf_data helper for major trend filter
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # 1d SMA(50) for major trend
        sma_1d = calculate_sma(c_1d, period=50)
        
        # Align 1d indicators to 4h timeframe (auto shift for completed bars)
        sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
        
    except Exception:
        sma_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # Z-score thresholds (avoid extreme entries)
    ZSCORE_MAX = 1.5
    ZSCORE_MIN = -1.5
    
    # MACD histogram threshold for momentum confirmation
    MACD_MIN = 0.0  # Just need positive for long, negative for short
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(100, 50 * 2, 30)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_4h[i]) or np.isnan(zscore_4h[i]) or atr_4h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned 1d SMA value
        sma_1d_val = sma_1d_aligned[i] if i < len(sma_1d_aligned) else 0
        
        # Check stoploss and take profit for existing positions FIRST
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            else:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_4h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_4h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_4h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_4h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_4h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_4h[i]
                    if price > trail_stop:
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
        
        # No existing position - check for new entry
        price = close[i]
        
        # 1d trend filter: price vs SMA(50)
        trend_1d = 0
        if sma_1d_val > 0:
            if price > sma_1d_val:
                trend_1d = 1
            elif price < sma_1d_val:
                trend_1d = -1
        
        # 4h Supertrend direction
        st_direction = st_trend_4h[i]
        
        # MACD momentum confirmation
        macd_ok_long = macd_hist[i] > MACD_MIN
        macd_ok_short = macd_hist[i] < -MACD_MIN
        
        # Z-score filter (avoid extreme overbought/oversold)
        zscore_ok_long = zscore_4h[i] < ZSCORE_MAX
        zscore_ok_short = zscore_4h[i] > ZSCORE_MIN
        
        # Entry logic: 1d trend + 4h Supertrend + MACD momentum + Z-score filter
        if trend_1d == 1 and st_direction == 1 and macd_ok_long and zscore_ok_long:
            # All conditions aligned for long
            signals[i] = SIZE_FULL
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            
        elif trend_1d == -1 and st_direction == -1 and macd_ok_short and zscore_ok_short:
            # All conditions aligned for short
            signals[i] = -SIZE_FULL
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 18:10
