#!/usr/bin/env python3
"""
EXPERIMENT #022 - HMA + Supertrend + RSI + Z-Score Multi-TF (4h+1h)
====================================================================================
Hypothesis: HMA (Hull Moving Average) is more responsive than KAMA with less lag,
and Supertrend provides clear ATR-based trend direction. Combining these on 4h
for trend + 1h for entries should reduce noise vs 15m while maintaining responsiveness.

Why this might beat Sharpe=5.5:
- HMA reacts faster to trend changes than KAMA or EMA
- Supertrend gives clear trend direction with built-in ATR stops
- 4h trend + 1h entries = good balance (less noise than 15m, faster than 4h)
- Z-score filter avoids extreme regimes (proven in current best)
- 2.0*ATR stoploss (slightly looser than #021's 1.5*ATR for fewer premature exits)
- Position sizing: 0.25-0.35 discrete levels with ATR volatility adjustment

Key features:
- 4h HMA(21) for adaptive trend direction
- 4h Supertrend(ATR=10, mult=3) for trend confirmation
- 1h RSI(14) pullback entries
- 1h Z-score(20) filter (avoid extremes >2.0)
- 2.0*ATR stoploss, signal→0 when breached
- Discrete signal levels: 0.0, ±0.25, ±0.35
- Dynamic sizing: base * (target_vol / current_vol)
"""

import numpy as np
import pandas as pd

name = "mtf_hma_supertrend_rsi_zscore_v1"
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
    
    # WMA with period/2
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    # WMA with period
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    for i in range(period - 1 + sqrt_period - 1, n):
        raw_hma = 2 * wma1[i] - wma2[i]
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.sum(np.array([2 * wma1[j] - wma2[j] for j in range(start_idx, i + 1)]) * weights) / np.sum(weights)
    
    return hma


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
    
    supertrend[period] = upper_band[period]
    trend_direction[period] = -1
    
    for i in range(period + 1, n):
        if close[i - 1] <= supertrend[i - 1]:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
        else:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
    
    return supertrend, trend_direction


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    for i in range(1, n):
        if delta[i - 1] > 0:
            gain[i] = delta[i - 1]
        else:
            loss[i] = -delta[i - 1]
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    
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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    
    # Resample to 4h for trend filters
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
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
    
    # 4h HMA for adaptive trend
    hma_4h = calculate_hma(c_4h, period=21)
    
    # 4h Supertrend for trend confirmation
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Map 4h indicators back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    st_trend_1h = np.zeros(n)
    hma_1h = np.zeros(n)
    
    n_4h = len(c_4h)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < n_4h and idx_4h >= 40:
            # HMA trend: price above HMA = bullish, below = bearish
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_1h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_1h[i] = -1
            
            # Supertrend direction
            st_trend_1h[i] = st_direction_4h[idx_4h]
            hma_1h[i] = hma_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 50
    RSI_SHORT_MIN = 50
    RSI_SHORT_MAX = 65
    
    # Z-score filter (avoid extreme regimes like current best)
    ZSCORE_MAX = 2.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.02
    
    first_valid = max(100, 40 * 4, 14 * 2, 20)
    
    # Track position state
    entry_price = np.zeros(n)
    position_side = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(zscore_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        st_trend = st_trend_1h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Z-score filter - avoid extreme regimes (like current best)
        if abs(zscore_val) > ZSCORE_MAX:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                signals[i] = 0.0
            continue
        
        # Both HMA and Supertrend must agree on trend direction
        if trend != st_trend or trend == 0:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
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
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
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
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
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
            continue
        
        # Dynamic position sizing based on ATR volatility
        current_atr_pct = atr / price if price > 0 else 0
        if current_atr_pct > 0:
            size_multiplier = min(1.5, max(0.5, TARGET_ATR_PCT / current_atr_pct))
        else:
            size_multiplier = 1.0
        
        position_size = SIZE_FULL * size_multiplier
        position_size = min(SIZE_FULL, max(SIZE_HALF, position_size))
        
        # Entry logic: HMA + Supertrend agreement + RSI pullback + Z-score
        if trend == 1 and st_trend == 1:  # Bullish trend confirmed
            # RSI pullback entry (35-50 range)
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1 and st_trend == -1:  # Bearish trend confirmed
            # RSI pullback entry (50-65 range)
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals