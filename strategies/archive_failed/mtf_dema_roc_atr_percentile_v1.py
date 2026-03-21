#!/usr/bin/env python3
"""
EXPERIMENT #018 - DEMA Crossover + ROC Momentum + ATR Volatility Filter
===============================================================================
Hypothesis: DEMA(8/21) crossover detects trend changes faster than Supertrend/HMA,
while ROC(10) momentum filter ensures we only enter when momentum confirms the trend.
ATR percentile filter avoids trading during extreme volatility regimes (crashes/spikes).
This should reduce whipsaw in choppy markets while capturing trends earlier.

Key innovations vs mtf_supertrend_macd_adx_v1:
- DEMA crossover instead of Supertrend for faster trend detection
- ROC(10) momentum confirmation instead of MACD (simpler, less lag)
- ATR percentile filter (20-80th percentile) instead of ADX
- Multi-timeframe: 4h DEMA trend + 1h ROC entries
- Discrete position sizing (0.0, ±0.25, ±0.35) with ATR trailing stop

Why this might beat Sharpe=1.278:
- DEMA responds 30-40% faster to trend changes than Supertrend
- ROC filter avoids entering when momentum is fading (common Supertrend weakness)
- ATR percentile avoids extreme volatility periods that cause large drawdowns
- Proven multi-timeframe approach from mtf_hma_rsi_zscore_v1 (Sharpe=5.4)
"""

import numpy as np
import pandas as pd

name = "mtf_dema_roc_atr_percentile_v1"
timeframe = "1h"
leverage = 1.0


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False).mean().values
    
    dema = 2 * ema1 - ema2
    dema[:period] = np.nan
    
    return dema


def calculate_roc(close, period=10):
    """Calculate Rate of Change for momentum confirmation"""
    n = len(close)
    roc = np.zeros(n)
    
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100
    
    return roc


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


def calculate_atr_percentile(atr, lookback=50):
    """Calculate ATR percentile to filter extreme volatility regimes"""
    n = len(atr)
    percentile = np.zeros(n)
    
    for i in range(lookback, n):
        window = atr[i - lookback + 1:i + 1]
        valid = window[window > 0]
        if len(valid) > 0:
            percentile[i] = np.sum(atr[i] >= valid) / len(valid) * 100
    
    return percentile


def calculate_rsi(close, period=14):
    """Calculate RSI for additional entry timing"""
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    atr_pct_1h = calculate_atr_percentile(atr_1h, lookback=50)
    roc_1h = calculate_roc(close, period=10)
    dema_fast_1h = calculate_dema(close, period=8)
    dema_slow_1h = calculate_dema(close, period=21)
    
    # 4h trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # 4h DEMA for trend direction
    dema_fast_4h = calculate_dema(c_4h, period=8)
    dema_slow_4h = calculate_dema(c_4h, period=21)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    
    # 4h trend direction based on DEMA crossover
    trend_4h = np.zeros(len(c_4h))
    for i in range(21, len(c_4h)):
        if np.isnan(dema_fast_4h[i]) or np.isnan(dema_slow_4h[i]):
            continue
        if dema_fast_4h[i] > dema_slow_4h[i]:
            trend_4h[i] = 1  # Bullish
        elif dema_fast_4h[i] < dema_slow_4h[i]:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.25   # Reduced position in marginal conditions
    
    # ROC thresholds for momentum confirmation
    ROC_LONG_MIN = 0.5    # Minimum ROC for long entry (positive momentum)
    ROC_SHORT_MAX = -0.5  # Maximum ROC for short entry (negative momentum)
    
    # ATR percentile filter - only trade in normal volatility
    ATR_PCT_MIN = 20   # Don't trade if ATR < 20th percentile (too quiet)
    ATR_PCT_MAX = 80   # Don't trade if ATR > 80th percentile (too volatile)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 50   # Enter long when RSI crosses above 50 in uptrend
    RSI_SHORT_ENTRY = 50  # Enter short when RSI crosses below 50 in downtrend
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(50, 21, 14, 10)  # Wait for all indicators
    
    # Track entry prices for trailing stop logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(roc_1h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(dema_fast_1h[i]) or np.isnan(dema_slow_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        roc_val = roc_1h[i]
        atr = atr_1h[i]
        price = close[i]
        atr_pct = atr_pct_1h[i]
        
        # ATR percentile filter - avoid extreme volatility regimes
        if atr_pct < ATR_PCT_MIN or atr_pct > ATR_PCT_MAX:
            if i > 0 and position_side[i - 1] != 0:
                # Hold existing position but don't add
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high relative to price
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_highest = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else price
            prev_lowest = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else price
            
            # Update highest/lowest since entry
            if prev_side == 1:
                highest_since_entry[i] = max(prev_highest, price)
                lowest_since_entry[i] = prev_lowest
                # Trailing stop for long: stop = highest - ATR_STOP_MULT * ATR
                stoploss_price = highest_since_entry[i] - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            elif prev_side == -1:
                lowest_since_entry[i] = min(prev_lowest, price)
                highest_since_entry[i] = prev_highest
                # Trailing stop for short: stop = lowest + ATR_STOP_MULT * ATR
                stoploss_price = lowest_since_entry[i] + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            
            # Also check initial stoploss from entry
            if prev_side == 1:
                initial_stop = prev_entry - ATR_STOP_MULT * atr
                if price < initial_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            elif prev_side == -1:
                initial_stop = prev_entry + ATR_STOP_MULT * atr
                if price > initial_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
        
        # 1h DEMA confirmation for entry timing
        dema_confirmed_long = dema_fast_1h[i] > dema_slow_1h[i]
        dema_confirmed_short = dema_fast_1h[i] < dema_slow_1h[i]
        
        if trend == 1:  # 4h uptrend
            # Need: 1h DEMA bullish + ROC positive + RSI confirmation
            if dema_confirmed_long and roc_val > ROC_LONG_MIN and rsi_val > RSI_LONG_ENTRY:
                # Strong momentum entry - full position
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif dema_confirmed_long and roc_val > 0 and rsi_val > 45:
                # Moderate momentum - half position
                signals[i] = SIZE_HALF
                position_side[i] = 1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    highest_since_entry[i] = highest_since_entry[i - 1]
                    lowest_since_entry[i] = lowest_since_entry[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # Need: 1h DEMA bearish + ROC negative + RSI confirmation
            if dema_confirmed_short and roc_val < ROC_SHORT_MAX and rsi_val < RSI_SHORT_ENTRY:
                # Strong momentum entry - full short
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif dema_confirmed_short and roc_val < 0 and rsi_val < 55:
                # Moderate momentum - half short
                signals[i] = -SIZE_HALF
                position_side[i] = -1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    highest_since_entry[i] = highest_since_entry[i - 1]
                    lowest_since_entry[i] = lowest_since_entry[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals