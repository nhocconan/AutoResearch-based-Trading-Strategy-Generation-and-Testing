#!/usr/bin/env python3
"""
EXPERIMENT #002 - MTF DEMA+Supertrend+RSI+Zscore (1h+4h v1)
==================================================================================================
Hypothesis: Switch to 1h primary timeframe (vs 15m) for cleaner signals with fewer trades.
Combine 4h Supertrend (trend filter) + 1h DEMA crossover (fast entry) + RSI pullback + Z-score regime.

Why this should work:
- 1h timeframe reduces noise vs 15m (fewer false signals, lower fee drag)
- DEMA (Double EMA) reacts faster than regular EMA for entry timing
- 4h Supertrend provides clear trend direction (proven in literature)
- Z-score filter avoids extreme mean-reversion conditions
- Fewer trades = lower fee impact (0.10% per signal change)

Position sizing: MAX 0.35, discrete levels (0.0, ±0.20, ±0.35)
Stoploss: 2*ATR trailing, Take profit: 2R reduce to half, trail at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_dema_supertrend_rsi_zscore_1h_4h_v1"
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


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False).mean().values
    
    dema = 2 * ema1 - ema2
    
    return dema


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
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price deviation)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(n):
        if std[i] > 0:
            zscore[i] = (close[i] - mean[i]) / std[i]
        else:
            zscore[i] = 0
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    dema_fast_1h = calculate_dema(close, period=8)
    dema_slow_1h = calculate_dema(close, period=21)
    
    # Get 4h data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h Supertrend for trend direction
        _, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
        rsi_4h = calculate_rsi(c_4h, period=14)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    except Exception:
        # Fallback if mtf_data fails
        st_direction_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n) + 50
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score filter (avoid extreme overbought/oversold)
    ZSCORE_MAX = 2.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(100, 21 * 2, 14 * 2, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned MTF values
        st_trend_4h = st_direction_4h_aligned[i] if i < len(st_direction_4h_aligned) else 0
        rsi_4h_val = rsi_4h_aligned[i] if i < len(rsi_4h_aligned) else 50
        
        # Z-score filter - avoid extreme conditions
        if abs(zscore_1h[i]) > ZSCORE_MAX:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 4h trend filter (Supertrend direction + RSI confirmation)
        # Long only if 4h Supertrend bullish AND 4h RSI not overbought
        # Short only if 4h Supertrend bearish AND 4h RSI not oversold
        trend_bullish = (st_trend_4h == 1 and rsi_4h_val < 70)
        trend_bearish = (st_trend_4h == -1 and rsi_4h_val > 30)
        
        if not trend_bullish and not trend_bearish:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                position_side[i] = 0
            else:
                position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
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
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered AND trend still valid
            if (prev_side == 1 and trend_bullish) or (prev_side == -1 and trend_bearish):
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                # Exit if trend reverses
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Entry logic: 4h trend + 1h DEMA crossover + RSI pullback
        price = close[i]
        
        # DEMA crossover signal
        dema_crossover_long = (dema_fast_1h[i] > dema_slow_1h[i]) and (dema_fast_1h[i-1] <= dema_slow_1h[i-1])
        dema_crossover_short = (dema_fast_1h[i] < dema_slow_1h[i]) and (dema_fast_1h[i-1] >= dema_slow_1h[i-1])
        
        if trend_bullish:
            # DEMA crossover OR RSI pullback entry
            if (dema_crossover_long or (dema_fast_1h[i] > dema_slow_1h[i])) and RSI_LONG_MIN <= rsi_1h[i] <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_bearish:
            # DEMA crossover OR RSI pullback entry
            if (dema_crossover_short or (dema_fast_1h[i] < dema_slow_1h[i])) and RSI_SHORT_MIN <= rsi_1h[i] <= RSI_SHORT_MAX:
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