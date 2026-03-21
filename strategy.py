#!/usr/bin/env python3
"""
EXPERIMENT #015 - MTF Daily Trend + 4h Supertrend + 1h RSI Pullback
==================================================================================================
Hypothesis: Simplify the multi-timeframe approach with clearer trend hierarchy.
- Daily SMA(50) = major trend direction (most reliable long-term filter)
- 4h Supertrend = intermediate trend confirmation
- 1h RSI = pullback entry timing (buy dips in uptrend, sell rallies in downtrend)
- Primary timeframe: 1h (fewer trades than 15m, cleaner signals, lower fees)

Why this should beat current best (Sharpe=0.065):
- Daily trend filter is more stable than 4h-only (reduces whipsaws)
- 1h base timeframe proven in #009 (Sharpe=0.065)
- Simpler logic = fewer bugs in position management
- Discrete signal levels (0.0, ±0.25, ±0.35) reduce churn costs

Key improvements over #004:
- Use 1h primary instead of 15m (fewer trades, better risk/reward)
- Add daily SMA(50) as ultimate trend filter (most reliable)
- Simplify position management (ATR stoploss only, no complex TP trailing)
- Fix position tracking bugs from #004
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_daily_sma_supertrend_rsi_1h_4h_1d_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Pad to match original length
    gain = np.pad(gain, (1, 0), mode='constant')
    loss = np.pad(loss, (1, 0), mode='constant')
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rsi = np.full(n, 50.0)
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


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
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    
    # Get 4h data using mtf_data helper
    df_4h = get_htf_data(prices, '4h')
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # 4h Supertrend for intermediate trend
    _, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    
    # Get 1d data using mtf_data helper
    df_1d = get_htf_data(prices, '1d')
    c_1d = df_1d['close'].values
    
    # Daily SMA(50) for major trend
    sma_1d = calculate_sma(c_1d, period=50)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Calculate daily trend (price vs SMA)
    trend_1d = np.zeros(n)
    for i in range(n):
        if i < len(c_1d) and i < len(sma_1d_aligned):
            if c_1d[min(i // 24, len(c_1d) - 1)] > sma_1d_aligned[i]:
                trend_1d[i] = 1
            elif c_1d[min(i // 24, len(c_1d) - 1)] < sma_1d_aligned[i]:
                trend_1d[i] = -1
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.08
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45  # Buy when RSI pulls back to 45 in uptrend
    RSI_SHORT_ENTRY = 55  # Sell when RSI rallies to 55 in downtrend
    RSI_EXIT = 50  # Exit when RSI crosses back through 50
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(100, 50 * 2)  # Need enough data for daily SMA
    
    # Track position state
    position = np.zeros(n)  # Current position size
    entry_price = np.zeros(n)
    stoploss_price = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            position[i] = 0
            continue
        
        # Get aligned MTF values
        st_4h = st_direction_4h_aligned[i] if i < len(st_direction_4h_aligned) else 0
        daily_trend = trend_1d[i]
        price = close[i]
        
        # Check existing position for stoploss
        if position[i - 1] != 0:
            prev_pos = position[i - 1]
            prev_entry = entry_price[i - 1]
            prev_stop = stoploss_price[i - 1]
            
            # Check stoploss
            if prev_pos > 0 and price < prev_stop:
                signals[i] = 0.0
                position[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                continue
            elif prev_pos < 0 and price > prev_stop:
                signals[i] = 0.0
                position[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                continue
            
            # Check RSI exit signal (momentum fading)
            if prev_pos > 0 and rsi_1h[i] > RSI_EXIT + 15:  # RSI > 65, overbought
                signals[i] = 0.0
                position[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                continue
            elif prev_pos < 0 and rsi_1h[i] < RSI_EXIT - 15:  # RSI < 35, oversold
                signals[i] = 0.0
                position[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                continue
            
            # Check trend reversal
            if prev_pos > 0 and (st_4h != 1 or daily_trend != 1):
                signals[i] = 0.0
                position[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                continue
            elif prev_pos < 0 and (st_4h != -1 or daily_trend != -1):
                signals[i] = 0.0
                position[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                continue
            
            # Hold position
            signals[i] = prev_pos
            position[i] = prev_pos
            entry_price[i] = prev_entry
            stoploss_price[i] = prev_stop
            continue
        
        # No position - look for entry
        # Long entry: Daily uptrend + 4h Supertrend up + RSI pullback
        if daily_trend == 1 and st_4h == 1:
            if RSI_LONG_ENTRY - 10 <= rsi_1h[i] <= RSI_LONG_ENTRY + 10:
                signals[i] = SIZE_FULL
                position[i] = SIZE_FULL
                entry_price[i] = price
                stoploss_price[i] = price - ATR_STOP_MULT * atr_1h[i]
                continue
        
        # Short entry: Daily downtrend + 4h Supertrend down + RSI rally
        elif daily_trend == -1 and st_4h == -1:
            if RSI_SHORT_ENTRY - 10 <= rsi_1h[i] <= RSI_SHORT_ENTRY + 10:
                signals[i] = -SIZE_FULL
                position[i] = -SIZE_FULL
                entry_price[i] = price
                stoploss_price[i] = price + ATR_STOP_MULT * atr_1h[i]
                continue
        
        # No signal
        signals[i] = 0.0
        position[i] = 0
    
    return signals