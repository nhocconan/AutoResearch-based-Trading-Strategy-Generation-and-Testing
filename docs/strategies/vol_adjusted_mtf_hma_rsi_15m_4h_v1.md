# Strategy: vol_adjusted_mtf_hma_rsi_15m_4h_v1

## Status
ACTIVE - Sharpe=0.058 | Return=+31.2% | DD=-32.8%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.532 | -10.6% | -34.7% | 337 |
| ETHUSDT | -0.320 | -5.9% | -37.8% | 306 |
| SOLUSDT | 1.027 | +109.9% | -26.1% | 8 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.856 | -3.9% | -14.1% | 191 |
| ETHUSDT | -0.343 | -1.2% | -24.0% | 41 |
| SOLUSDT | -0.221 | -0.2% | -20.5% | 57 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #056 - Volatility-Adjusted MTF with Simple Signal Logic (15m + 4h)
==================================================================================================
Hypothesis: Ensemble strategies (#051-054) failed due to signal conflicts causing excessive churn.
This strategy uses SIMPLER signal logic with volatility-adjusted position sizing to reduce fees
and improve risk-adjusted returns.

Key improvements over #055:
- Simpler signal logic: 4h HMA trend + 15m RSI pullback ONLY (fewer conditions = fewer signal changes)
- Volatility-adjusted sizing: position_size = base_size * (target_vol / current_ATR_pct)
- Discrete signal levels: 0.0, ±0.20, ±0.30, ±0.35 (minimize churn costs)
- Tighter stoploss: 2.0*ATR instead of 2.5*ATR (faster exit on wrong trades)
- Mandatory 4h trend filter (no trades against 4h trend)
- Reduced signal frequency: only enter on RSI extremes (30/70) not ranges

Why this should beat #055:
- Fewer signal changes = lower fees (0.10% per change)
- Volatility sizing = consistent risk per trade across market conditions
- Simpler logic = more robust across BTC/ETH/SOL
- Tighter stops = smaller losses on failed trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "vol_adjusted_mtf_hma_rsi_15m_4h_v1"
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
    
    wma1 = pd.Series(close).rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    ).values
    
    raw_hma = 2 * wma1 - wma2
    
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    ).values
    
    return np.nan_to_num(hma, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100.0
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0
    
    return rsi


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.ones(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    supertrend[period - 1] = lower_band[period - 1]
    
    for i in range(period, n):
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    _, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # Get 4h data using mtf_data helper (CRITICAL for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators for trend
        hma_4h = calculate_hma(close_4h, period=21)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        
        # Align 4h indicators to 15m timeframe
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_4h_aligned = calculate_hma(close, period=48)
        atr_4h_aligned = calculate_atr(high, low, close, period=14)
    
    # Generate signals with volatility-adjusted sizing
    signals = np.zeros(n)
    
    # Base position sizes - DISCRETE levels to minimize churn
    SIZE_LOW = 0.20
    SIZE_MED = 0.30
    SIZE_HIGH = 0.35  # MAX - critical for drawdown control
    
    # RSI thresholds for entry (extremes only to reduce churn)
    RSI_LONG_ENTRY = 35
    RSI_SHORT_ENTRY = 65
    RSI_EXIT = 50
    
    # ATR stoploss multiplier (tighter than #055)
    ATR_STOP_MULT = 2.0
    ATR_TP_MULT = 2.0  # 2R take profit
    
    # Volatility target for position sizing
    TARGET_VOL_PCT = 0.02  # Target 2% daily volatility
    
    first_valid = max(100, 48 * 4)  # Need enough 4h bars
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] <= 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or hma_4h_aligned[i] <= 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 4h trend direction (mandatory filter)
        if close[i] > hma_4h_aligned[i]:
            trend_4h = 1
        elif close[i] < hma_4h_aligned[i]:
            trend_4h = -1
        else:
            trend_4h = 0
        
        rsi_val = rsi_15m[i]
        price = close[i]
        atr = atr_15m[i]
        
        # Calculate volatility-adjusted position size
        # ATR as % of price
        atr_pct = atr / price if price > 0 else 0.01
        
        # Scale position: higher vol = smaller position
        # target_vol / current_vol * base_size
        if atr_pct > 0:
            vol_scale = min(2.0, max(0.5, TARGET_VOL_PCT / atr_pct))
        else:
            vol_scale = 1.0
        
        # Check stoploss and take profit for existing positions FIRST
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
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + ATR_TP_MULT * atr
                if not prev_tp and price >= tp_price:
                    # Reduce to half position
                    base_size = SIZE_LOW * vol_scale
                    signals[i] = prev_side * base_size * 0.5
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit after TP hit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - ATR_TP_MULT * atr
                if not prev_tp and price <= tp_price:
                    base_size = SIZE_LOW * vol_scale
                    signals[i] = prev_side * base_size * 0.5
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit after TP hit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
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
        
        # Check for 4h trend reversal - exit if trend changes
        if trend_4h == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # ENTRY LOGIC - Simple: RSI extreme + 4h trend alignment
        signal_direction = 0
        signal_strength = 0
        
        # LONG entry: 4h bullish + RSI oversold pullback
        if trend_4h == 1:
            if rsi_val <= RSI_LONG_ENTRY:
                signal_direction = 1
                signal_strength = SIZE_HIGH
            elif rsi_val <= RSI_LONG_ENTRY + 10 and st_direction_15m[i] == 1:
                signal_direction = 1
                signal_strength = SIZE_MED
        
        # SHORT entry: 4h bearish + RSI overbought bounce
        elif trend_4h == -1:
            if rsi_val >= RSI_SHORT_ENTRY:
                signal_direction = -1
                signal_strength = SIZE_HIGH
            elif rsi_val >= RSI_SHORT_ENTRY - 10 and st_direction_15m[i] == -1:
                signal_direction = -1
                signal_strength = SIZE_MED
        
        # Apply volatility scaling
        if signal_direction != 0:
            position_size = signal_strength * vol_scale
            position_size = min(SIZE_HIGH, max(0.0, position_size))  # Clamp to max
            
            # Discretize to reduce churn
            if position_size >= 0.32:
                position_size = SIZE_HIGH
            elif position_size >= 0.25:
                position_size = SIZE_MED
            elif position_size >= 0.15:
                position_size = SIZE_LOW
            else:
                position_size = 0.0
            
            if position_size > 0:
                signals[i] = signal_direction * position_size
                position_side[i] = signal_direction
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 14:13
