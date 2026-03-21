#!/usr/bin/env python3
"""
EXPERIMENT #018 - Supertrend + Bollinger Squeeze + RSI + ATR Stop
=================================================================
Hypothesis: Combining 4h Supertrend (trend direction) with 1h Bollinger 
squeeze detection (low volatility before breakout) and RSI pullback entries
should capture momentum moves with better risk/reward than pure MA strategies.

Key innovations vs mtf_donchian_hma_rsi_zscore_v1:
- Supertrend(4h) instead of Donchian for volatility-adjusted trend
- Bollinger Band Width squeeze filter (BW < 20th percentile) for entry timing
- RSI(14) pullback in trend direction for entry confirmation
- ATR(14) trailing stop at 2.5*ATR with proper position tracking
- Discrete position sizing (0.0, ±0.25, ±0.35) to minimize churn costs

Why this might beat Sharpe=2.139:
- Supertrend adapts to volatility better than fixed Donchian channels
- Bollinger squeeze identifies low-volatility compression before breakouts
- Multi-timeframe logic (4h trend + 1h entry) proven in prior experiments
- Conservative position sizing (max 0.35) controls drawdown
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_bb_squeeze_rsi_v1"
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
    """Calculate Supertrend indicator for trend direction"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if atr[i] == 0:
            continue
            
        upper_band = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            trend[i] = -1 if close[i] < supertrend[i] else 1
        else:
            # Update bands based on trend
            if trend[i - 1] == 1:
                supertrend[i] = max(lower_band, supertrend[i - 1])
            else:
                supertrend[i] = min(upper_band, supertrend[i - 1])
            
            # Check for trend reversal
            if close[i] > supertrend[i] and trend[i - 1] == -1:
                trend[i] = 1
                supertrend[i] = lower_band
            elif close[i] < supertrend[i] and trend[i - 1] == 1:
                trend[i] = -1
                supertrend[i] = upper_band
            else:
                trend[i] = trend[i - 1]
    
    return supertrend, trend


def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    
    return upper, lower, sma, bandwidth


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
    mask = (avg_loss > 0) & (avg_gain >= 0)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[avg_loss == 0] = 100
    
    return rsi


def calculate_bw_percentile(bandwidth, lookback=100):
    """Calculate Bollinger Band Width percentile for squeeze detection"""
    n = len(bandwidth)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bandwidth[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(bandwidth[i] >= valid) / len(valid)
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger(close, period=20, std_mult=2.0)
    bw_percentile = calculate_bw_percentile(bb_bandwidth, lookback=100)
    
    # 4h Supertrend for trend filter (resample 1h → 4h)
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
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Calculate 4h Supertrend
    _, trend_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
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
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    
    # Bollinger squeeze threshold (percentile < 0.3 = low volatility)
    BB_SQUEEZE_THRESHOLD = 0.30
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(100, 20, 14, 40)  # Wait for all indicators
    
    # Track entry prices for trailing stop logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bb_bandwidth[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        bw_pct = bw_percentile[i]
        
        # ATR filter - avoid trading when ATR is extremely high
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
                # Trailing stop for long
                stoploss_price = max(prev_entry, highest_since_entry[i] - ATR_STOP_MULT * atr) - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            elif prev_side == -1:
                highest_since_entry[i] = prev_highest
                lowest_since_entry[i] = min(prev_lowest, price)
                # Trailing stop for short
                stoploss_price = min(prev_entry, lowest_since_entry[i] + ATR_STOP_MULT * atr) + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            else:
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
        else:
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
        
        # Bollinger squeeze filter - only enter when volatility is compressed
        is_squeeze = bw_pct < BB_SQUEEZE_THRESHOLD
        
        if trend == 1:  # 4h uptrend
            # RSI pullback entry with Bollinger squeeze confirmation
            if rsi_val < RSI_LONG_ENTRY:
                if is_squeeze:
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                else:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = price
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
            elif rsi_val < 50:
                # Moderate pullback - hold or reduce
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    highest_since_entry[i] = highest_since_entry[i - 1]
                    lowest_since_entry[i] = lowest_since_entry[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                # RSI too high - hold existing or stay flat
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    highest_since_entry[i] = highest_since_entry[i - 1]
                    lowest_since_entry[i] = lowest_since_entry[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # RSI rally entry with Bollinger squeeze confirmation
            if rsi_val > RSI_SHORT_ENTRY:
                if is_squeeze:
                    signals[i] = -SIZE_FULL
                    position_side[i] = -1
                    entry_price[i] = price
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                else:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = price
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
            elif rsi_val > 50:
                # Moderate rally - hold or reduce
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    highest_since_entry[i] = highest_since_entry[i - 1]
                    lowest_since_entry[i] = lowest_since_entry[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                # RSI too low - hold existing or stay flat
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    highest_since_entry[i] = highest_since_entry[i - 1]
                    lowest_since_entry[i] = lowest_since_entry[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
    
    return signals