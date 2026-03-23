# Strategy: simplified_mtf_trend_volume_15m_1h_v1

## Status
ACTIVE - Sharpe=0.159 | Return=+62.0% | DD=-29.4%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.884 | -17.5% | -33.3% | 400 |
| ETHUSDT | 0.272 | +35.6% | -20.4% | 2 |
| SOLUSDT | 1.090 | +167.8% | -34.4% | 18 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -2.318 | -13.4% | -18.3% | 283 |
| ETHUSDT | -0.533 | -2.0% | -16.4% | 131 |
| SOLUSDT | -0.246 | +1.0% | -18.4% | 104 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #076 - Simplified MTF Trend-Follow with Volume Confirmation
==================================================================================================
Hypothesis: Previous ensemble strategies suffered from excessive complexity and churn.
This version simplifies to 2 core indicators (HMA + Supertrend) with volume confirmation,
using 1h trend filter (more responsive than 4h for 15m entries).

Key innovations:
1. SIMPLER: 2 indicators instead of 3 (HMA + Supertrend) - less noise
2. 1h trend filter instead of 4h - more responsive for 15m entries
3. Volume confirmation - only enter on above-average volume bars
4. Discrete signal levels (0.0, ±0.30) - minimize churn costs
5. Tighter stoploss (1.5*ATR) with trailing at 1R
6. Only flip signals on confirmed trend change (reduce whipsaws)

Why this should beat current best (Sharpe=3.653):
- Less churn = fewer fees eating profits
- Volume filter avoids false breakouts
- 1h trend more responsive than 4h for 15m timeframe
- Based on #075 learnings but simplified
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "simplified_mtf_trend_volume_15m_1h_v1"
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
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50
    
    return rsi


def calculate_volume_sma(volume, period=20):
    """Calculate volume simple moving average"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


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
            zscore[i] = 0
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # ========== 15m indicators for entry timing ==========
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    _, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    vol_sma_15m = calculate_volume_sma(volume, period=20)
    
    # ========== 1h indicators via mtf_data helper (CRITICAL) ==========
    try:
        df_1h = get_htf_data(prices, '1h')
        close_1h = df_1h['close'].values
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        
        # Calculate 1h indicators
        hma_1h = calculate_hma(close_1h, period=21)
        _, st_direction_1h = calculate_supertrend(high_1h, low_1h, close_1h, period=10, multiplier=3.0)
        
        # Align 1h indicators to 15m timeframe (auto shift for completed bars)
        hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
        st_1h_aligned = align_htf_to_ltf(prices, df_1h, st_direction_1h)
        
    except Exception as e:
        # Fallback if mtf_data fails
        hma_1h_aligned = hma_15m
        st_1h_aligned = st_direction_15m
    
    # ========== Generate signals ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_LONG = 0.30  # Conservative position size
    SIZE_SHORT = -0.30
    SIZE_HALF_LONG = 0.15
    SIZE_HALF_SHORT = -0.15
    
    # Entry thresholds
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 55
    ZSCORE_MAX = 2.0
    ATR_STOP_MULT = 1.5  # Tighter stoploss
    VOLUME_MULT = 1.2  # Volume must be 20% above average
    
    first_valid = max(200, 100, 40)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    prev_signal = 0.0
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # ========== 1h Trend Filter (need both HMA and Supertrend agreement) ==========
        hma_trend_1h = 1 if close[i] > hma_1h_aligned[i] else (-1 if close[i] < hma_1h_aligned[i] else 0)
        st_trend_1h = st_1h_aligned[i]
        
        # Volume filter
        vol_ratio = volume[i] / vol_sma_15m[i] if vol_sma_15m[i] > 0 else 1.0
        volume_confirmed = vol_ratio >= VOLUME_MULT
        
        # ========== Check existing positions first ==========
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            atr = atr_15m[i]
            price = close[i]
            
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
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    prev_signal = 0.0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF_LONG
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        prev_signal = 0.0
                        continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    prev_signal = 0.0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = SIZE_HALF_SHORT
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        prev_signal = 0.0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== New Entry Logic ==========
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        st_15m = st_direction_15m[i]
        
        # Only enter on volume confirmation for breakouts
        # For pullback entries, volume less critical
        
        # LONG entry: 1h trend bullish (both HMA and Supertrend agree)
        if hma_trend_1h == 1 and st_trend_1h == 1:
            # Check 15m Supertrend also bullish
            if st_15m == 1:
                # RSI pullback entry
                if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and abs(zscore_val) < ZSCORE_MAX:
                    signals[i] = SIZE_LONG
                    position_side[i] = 1
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
                    prev_signal = SIZE_LONG
        
        # SHORT entry: 1h trend bearish (both HMA and Supertrend agree)
        elif hma_trend_1h == -1 and st_trend_1h == -1:
            # Check 15m Supertrend also bearish
            if st_15m == -1:
                # RSI pullback entry
                if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and abs(zscore_val) < ZSCORE_MAX:
                    signals[i] = SIZE_SHORT
                    position_side[i] = -1
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
                    prev_signal = SIZE_SHORT
        
        # Default: no position
        if signals[i] == 0:
            position_side[i] = 0
            prev_signal = 0.0
    
    return signals
```

## Last Updated
2026-03-21 14:39
