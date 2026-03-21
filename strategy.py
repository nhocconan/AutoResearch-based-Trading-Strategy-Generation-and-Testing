#!/usr/bin/env python3
"""
EXPERIMENT #010 - MTF EMA Trend + RSI Pullback + Stochastic Entry 1h/4h v1
==================================================================================================
Hypothesis: Current #009 uses Supertrend+KAMA+RSI+Volume but has low Sharpe (0.029).
The winning #004 used Supertrend+MACD+BBW+RSI across 15m/1h/4h with Sharpe=3.653.

Key changes from #009:
- Trend: 4h EMA(21/55) crossover (smoother than Supertrend, less whipsaw)
- Entry: 1h RSI pullback (40-60 range) + Stochastic %D confirmation
- Filter: 1h Bollinger Band width percentile (avoid low volatility squeezes)
- Remove: KAMA, Volume, ADX (too many filters reduced trade count in #009)
- Stoploss: 2.5*ATR from entry (same as #009, proven working)
- Position size: 0.35 (proven safe across all experiments)

Why this should beat #009:
- EMA crossover is smoother than Supertrend for trend direction
- Stochastic adds momentum confirmation without overfiltering
- BB width filter avoids entering during dead markets (major issue in #009)
- 1h timeframe balances noise reduction vs trade frequency better than 15m
- Based on #004 winning formula but with cleaner indicator stack
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_ema_rsi_stochastic_bbwidth_1h_4h_v1"
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


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator %K and %D"""
    n = len(close)
    if n < k_period:
        return np.zeros(n), np.zeros(n)
    
    k = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest = np.min(low[i - k_period + 1:i + 1])
        highest = np.max(high[i - k_period + 1:i + 1])
        
        if highest > lowest:
            k[i] = 100 * (close[i] - lowest) / (highest - lowest)
        else:
            k[i] = 50
    
    for i in range(k_period - 1 + d_period - 1, n):
        d[i] = np.mean(k[i - d_period + 1:i + 1])
    
    return k, d


def calculate_bollinger_bands(close, period=20, multiplier=2.0):
    """Calculate Bollinger Bands and bandwidth"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + multiplier * std
    lower = middle - multiplier * std
    bandwidth = (upper - lower) / middle
    
    return upper, lower, middle, bandwidth


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    stoch_k_1h, stoch_d_1h = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    _, _, _, bb_width_1h = calculate_bollinger_bands(close, period=20, multiplier=2.0)
    
    # Calculate BB width percentile for regime filter
    bb_width_percentile = pd.Series(bb_width_1h).rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x < x[-1]) / len(x) if len(x) > 0 else 0.5, raw=True
    ).values
    
    # 4h trend filter using mtf_data helper (MANDATORY)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        
        # Calculate 4h EMA crossover (21/55)
        ema_21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21).mean().values
        ema_55_4h = pd.Series(close_4h).ewm(span=55, min_periods=55).mean().values
        
        # EMA trend direction: 1 if EMA21 > EMA55, -1 if EMA21 < EMA55, 0 otherwise
        trend_4h_raw = np.zeros(len(close_4h))
        for i in range(len(close_4h)):
            if ema_21_4h[i] > ema_55_4h[i]:
                trend_4h_raw[i] = 1
            elif ema_21_4h[i] < ema_55_4h[i]:
                trend_4h_raw[i] = -1
            else:
                trend_4h_raw[i] = 0
        
        # Align 4h trend to 1h timeframe (auto shift for completed bars)
        trend_4h = align_htf_to_ltf(prices, df_4h, trend_4h_raw)
    except Exception:
        # Fallback if mtf_data fails
        trend_4h = np.ones(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Stochastic %D thresholds for momentum confirmation
    STOCH_LONG_MAX = 50  # %D below 50 for long entry (pullback)
    STOCH_SHORT_MIN = 50  # %D above 50 for short entry (pullback)
    
    # BB width percentile threshold (avoid low volatility)
    BB_WIDTH_PERCENTILE_MIN = 0.30
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 55 * 2, 100)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    entry_atr = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        trend = trend_4h[i]
        rsi_val = rsi_1h[i]
        stoch_d = stoch_d_1h[i]
        atr = atr_1h[i]
        price = close[i]
        bb_percentile = bb_width_percentile[i]
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_atr = entry_atr[i - 1] if entry_atr[i - 1] > 0 else atr
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
            
            # Stoploss check (2.5*ATR from ENTRY)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * prev_atr
                if low[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * prev_atr
                if not prev_tp and high[i] >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    entry_atr[i] = prev_atr
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * prev_atr
                    if low[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        entry_atr[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * prev_atr
                if high[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * prev_atr
                if not prev_tp and low[i] <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    entry_atr[i] = prev_atr
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * prev_atr
                    if high[i] > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        entry_atr[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            entry_atr[i] = entry_atr[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h EMA trend + 1h RSI pullback + Stochastic + BB width filter
        # BB width filter - avoid low volatility regimes
        if np.isnan(bb_percentile) or bb_percentile < BB_WIDTH_PERCENTILE_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        if trend == 1:  # Bullish trend on 4h
            # RSI pullback entry (not overbought, not oversold)
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                # Stochastic confirmation: %D below 50 (pullback in uptrend)
                if stoch_d <= STOCH_LONG_MAX:
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    entry_atr[i] = atr
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                
        elif trend == -1:  # Bearish trend on 4h
            # RSI pullback entry (not overbought, not oversold)
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                # Stochastic confirmation: %D above 50 (pullback in downtrend)
                if stoch_d >= STOCH_SHORT_MIN:
                    signals[i] = -SIZE_FULL
                    position_side[i] = -1
                    entry_price[i] = price
                    entry_atr[i] = atr
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals