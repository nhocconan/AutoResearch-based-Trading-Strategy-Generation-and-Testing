# Strategy: simplified_mtf_trend_1h_4h_v3

## Status
ACTIVE - Sharpe=0.047 | Return=+57.9% | DD=-29.9%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -1.048 | -22.8% | -33.6% | 112 |
| ETHUSDT | 0.057 | +21.8% | -22.2% | 45 |
| SOLUSDT | 1.133 | +174.8% | -33.9% | 8 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -1.123 | -4.6% | -12.2% | 75 |
| ETHUSDT | -1.009 | -8.6% | -25.3% | 75 |
| SOLUSDT | -0.727 | -7.4% | -16.1% | 61 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #088 - SIMPLIFIED MTF TREND FOLLOWING 1H_4H_V3
==================================================================================================
Hypothesis: Simplify the ensemble approach to reduce bugs while keeping multi-timeframe edge.

Key innovations:
- 4h HMA trend filter (cleaner than voting)
- 1h RSI pullback entries in direction of 4h trend
- ATR-based trailing stops with take-profit scaling
- Regime filter via BBW percentile (trend vs mean-revert mode)
- Discrete position sizes: 0.0, 0.20, 0.35 (reduces churn costs)

Why this should work:
- Simpler than #087 voting logic (fewer bug opportunities)
- 4h trend filter keeps us on right side of major moves
- RSI pullbacks give better entry timing than pure trend follow
- ATR stops protect against large drawdowns
- Discrete sizing reduces fee drag from signal churn

Risk Management:
- Max position: 0.35 (35% of capital)
- Stop loss: 2.5x ATR from entry
- Take profit: reduce to half at 2R, trail stop at 1R
- No leverage (leverage=1.0)
"""

import numpy as np
import pandas as pd

name = "simplified_mtf_trend_1h_4h_v3"
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        raw_vals = 2 * wma1[start_idx:i + 1] - wma2[start_idx:i + 1]
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    signals = np.zeros(n)
    
    # 1h indicators for entry
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    _, _, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_1h = calculate_bbw_percentile(bbw_1h, lookback=100)
    
    # 4h trend filter (downsample 1h → 4h: 4 bars per 4h)
    bars_per_4h = 4
    n_4h = n // bars_per_4h
    
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * bars_per_4h
        end_idx = start_idx + bars_per_4h
        c_4h[i] = close[end_idx - 1]
        h_4h[i] = np.max(high[start_idx:end_idx])
        l_4h[i] = np.min(low[start_idx:end_idx])
    
    # 4h indicators for trend
    hma_4h = calculate_hma(c_4h, period=21)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    
    # Map 4h indicators back to 1h
    trend_4h = np.zeros(n)
    regime_4h = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 21:
            # 4h trend direction
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            # Regime: low BBW pct = trending, high BBW pct = mean-reverting
            if bbw_pct_4h[idx_4h] < 0.3:
                regime_4h[i] = 1  # Trend regime
            elif bbw_pct_4h[idx_4h] > 0.7:
                regime_4h[i] = -1  # Mean-reversion regime
            else:
                regime_4h[i] = 0  # Neutral
    
    # Position sizing - DISCRETE levels
    SIZE_BASE = 0.20
    SIZE_MAX = 0.35
    SIZE_HALF = 0.15
    
    # Thresholds
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 50
    RSI_SHORT_MIN = 50
    RSI_SHORT_MAX = 65
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 100 + 21, 14 * 2, 20)
    
    # Track position state using lists (avoid read-only array issues)
    position_side = [0] * n
    entry_price = [0.0] * n
    tp_triggered = [False] * n
    highest_since_entry = [0.0] * n
    lowest_since_entry = [0.0] * n
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        rsi_val = rsi_1h[i]
        trend = trend_4h[i]
        regime = regime_4h[i]
        
        # Check existing positions first (stoploss/TP)
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0.0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0.0
                    lowest_since_entry[i] = 0.0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop after TP
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0.0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0.0
                        lowest_since_entry[i] = 0.0
                        continue
            
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0.0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0.0
                    lowest_since_entry[i] = 0.0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop after TP
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0.0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0.0
                        lowest_since_entry[i] = 0.0
                        continue
            
            # Hold position - no signal change
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # NEW ENTRY LOGIC
        signal_long = False
        signal_short = False
        
        # Regime-adaptive entry logic
        if regime == 1:  # Trend regime - follow 4h trend on RSI pullback
            if trend == 1 and RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                signal_long = True
            elif trend == -1 and RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                signal_short = True
        elif regime == -1:  # Mean-reversion regime - fade extremes
            if rsi_val < RSI_LONG_MIN:
                signal_long = True
            elif rsi_val > RSI_SHORT_MAX:
                signal_short = True
        else:  # Neutral regime - require stronger trend confirmation
            if trend == 1 and rsi_val < 50:
                signal_long = True
            elif trend == -1 and rsi_val > 50:
                signal_short = True
        
        # Determine position size based on regime confidence
        if signal_long:
            size = SIZE_MAX if regime == 1 else SIZE_BASE
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        elif signal_short:
            size = SIZE_MAX if regime == 1 else SIZE_BASE
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0.0
            tp_triggered[i] = False
            highest_since_entry[i] = 0.0
            lowest_since_entry[i] = 0.0
    
    return signals
```

## Last Updated
2026-03-21 10:35
