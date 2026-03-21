#!/usr/bin/env python3
"""
EXPERIMENT #040 - Supertrend Trend + RSI+Z-score Entries + Trailing Stop
====================================================================================
Hypothesis: Replace HMA with Supertrend for more robust trend detection in crypto.
Supertrend adapts to volatility via ATR, providing clearer trend signals than HMA.
Combine 4h Supertrend trend with 1h RSI+Z-score entries for better filter.
Add trailing stop that activates after 1R profit to lock gains.

Key improvements over #021:
- Supertrend(10,3) instead of HMA - volatility-adaptive trend
- 1h entries instead of 15m - less noise, fewer whipsaws
- Z-score filter on entries - avoid extremes
- Trailing stop after 1R profit - lock in gains progressively
- More conservative sizing: 0.25-0.30 max (vs 0.35)
- Better signal continuity - reduce churn costs

Why this might beat Sharpe=11.523:
- Supertrend is more robust in crypto's volatile regimes
- 1h entries reduce false signals from 15m noise
- Z-score + RSI dual filter improves entry quality
- Trailing stop captures more of trending moves
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_rsi_zscore_trail_1h_v1"
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
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for bullish, -1 for bearish
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band = mid + multiplier * atr[i]
        lower_band = mid - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            if direction[i - 1] == 1:
                if close[i] < lower_band:
                    direction[i] = -1
                    supertrend[i] = upper_band
                else:
                    direction[i] = 1
                    supertrend[i] = max(lower_band, supertrend[i - 1])
            else:
                if close[i] > upper_band:
                    direction[i] = 1
                    supertrend[i] = lower_band
                else:
                    direction[i] = -1
                    supertrend[i] = min(upper_band, supertrend[i - 1])
    
    return supertrend, direction


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


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = std > 0
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    supertrend_1h, direction_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
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
    supertrend_4h, direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(direction_4h):
            trend_1h[i] = direction_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.30   # Full position (more conservative than 0.35)
    SIZE_HALF = 0.15   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # Z-score thresholds for regime filter
    ZSCORE_MAX = 2.0      # Don't enter if price > 2.0 std dev from mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # Trailing stop activation (after 1R profit)
    TRAIL_ACTIVATION = 1.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.02  # Target 2% ATR
    
    first_valid = max(80, 40, 14, 20)  # Wait for all indicators
    
    # Track position state
    position_side = np.zeros(n, dtype=int)  # 1 for long, -1 for short, 0 for flat
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=int)  # Track if take profit was hit
    trailing_active = np.zeros(n, dtype=int)  # Track if trailing stop is active
    
    for i in range(first_valid, n):
        # Initialize from previous state
        if i > 0:
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            trailing_active[i] = trailing_active[i - 1]
            signals[i] = signals[i - 1]
        
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]):
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            trailing_active[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if position_side[i] != 0:
            prev_side = position_side[i]
            prev_entry = entry_price[i] if entry_price[i] > 0 else price
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i], price)
                
                # Check if we should activate trailing stop (after 1R profit)
                if not trailing_active[i]:
                    profit_r = (price - prev_entry) / (ATR_STOP_MULT * atr) if atr > 0 else 0
                    if profit_r >= TRAIL_ACTIVATION:
                        trailing_active[i] = 1
                
                # Trailing stop logic
                if trailing_active[i]:
                    trail_stop = highest_since_entry[i] - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        trailing_active[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # Hard stoploss check
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_active[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not tp_triggered[i] and price >= tp_price:
                    # Reduce to half position
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    tp_triggered[i] = 1
                    trailing_active[i] = 1  # Activate trailing after TP
                    continue
                
                # RSI exit signal
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_active[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                    
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i], price)
                
                # Check if we should activate trailing stop (after 1R profit)
                if not trailing_active[i]:
                    profit_r = (prev_entry - price) / (ATR_STOP_MULT * atr) if atr > 0 else 0
                    if profit_r >= TRAIL_ACTIVATION:
                        trailing_active[i] = 1
                
                # Trailing stop logic
                if trailing_active[i]:
                    trail_stop = lowest_since_entry[i] + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        trailing_active[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # Hard stoploss check
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_active[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not tp_triggered[i] and price <= tp_price:
                    # Reduce to half position
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    tp_triggered[i] = 1
                    trailing_active[i] = 1  # Activate trailing after TP
                    continue
                
                # RSI exit signal
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    trailing_active[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
        
        # Z-score filter - avoid entering at extreme deviations
        if abs(zscore_val) > ZSCORE_MAX:
            if position_side[i] != 0:
                # Hold existing position
                continue
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
            # RSI pullback entry in uptrend
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30 and position_side[i] == 0:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                trailing_active[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif position_side[i] == 0:
                signals[i] = 0.0
                    
        elif trend == -1:  # 4h downtrend
            # RSI rally entry in downtrend
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70 and position_side[i] == 0:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                trailing_active[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif position_side[i] == 0:
                signals[i] = 0.0
        else:  # No clear trend
            if position_side[i] == 0:
                signals[i] = 0.0
    
    return signals