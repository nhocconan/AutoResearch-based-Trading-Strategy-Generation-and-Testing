#!/usr/bin/env python3
"""
EXPERIMENT #017 - KAMA Adaptive Trend + MACD Momentum + ADX Strength Filter
===============================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise better 
than HMA/EMA, reducing whipsaw in choppy conditions. Combined with MACD histogram 
cross for entry timing and ADX strength filter (>25), this should capture strong 
trends while avoiding false signals in ranging markets.

Key improvements vs mtf_donchian_hma_rsi_zscore_v1:
- KAMA adapts smoothing based on market efficiency (faster in trends, slower in noise)
- MACD histogram cross provides clearer momentum entry signals than RSI pullback
- ADX filter ensures we only trade when trend strength is sufficient
- Simpler position tracking without complex entry_price arrays
- Faster computation (no multi-timeframe resampling overhead)
- Discrete signal levels (0.0, ±0.25, ±0.35) to minimize churn costs

Why this might beat Sharpe=2.139:
- KAMA proven effective in crypto volatility (adapts to regime changes)
- MACD + ADX combination filters out weak trends that cause drawdown
- Simpler logic = fewer bugs and faster backtest execution
- Position sizing capped at 0.35 to control drawdown during crashes
"""

import numpy as np
import pandas as pd

name = "mtf_kama_macd_adx_atr_v2"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio
    """
    n = len(close)
    kama = np.zeros(n)
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
    
    # Calculate smoothing constant
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    # Calculate KAMA
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength"""
    n = len(close)
    adx = np.zeros(n)
    
    if n < period * 2:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth using Wilder's method
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    atr[period - 1] = np.mean(tr[1:period])
    plus_di[period - 1] = 100 * np.mean(plus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    minus_di[period - 1] = 100 * np.mean(minus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_di[i] = 100 * ((plus_di[i - 1] * (period - 1)) + plus_dm[i]) / (period * atr[i]) if atr[i] > 0 else 0
        minus_di[i] = 100 * ((minus_di[i - 1] * (period - 1)) + minus_dm[i]) / (period * atr[i]) if atr[i] > 0 else 0
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Calculate all indicators
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    macd_line, signal_line, histogram = calculate_macd(close, fast=12, slow=26, signal=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Position sizing - DISCRETE levels to reduce churn costs
    SIZE_FULL = 0.35   # Full position in strong trend
    SIZE_HALF = 0.25   # Reduced position in moderate trend
    
    # Thresholds
    ADX_STRONG = 25     # Only trade when ADX > 25 (strong trend)
    ADX_MODERATE = 20   # Hold position if ADX > 20
    ATR_STOP_MULT = 2.5  # Stoploss at 2.5 * ATR
    
    # Wait for all indicators to be valid
    first_valid = max(40, 35, 28, 14)  # KAMA + MACD + ADX + ATR warmup
    
    signals = np.zeros(n)
    
    # Track position state for stoploss
    position_side = 0  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_atr = 0.0
    
    for i in range(first_valid, n):
        # Check for NaN values
        if np.isnan(kama[i]) or np.isnan(macd_line[i]) or np.isnan(histogram[i]) or np.isnan(adx[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        price = close[i]
        current_adx = adx[i]
        current_hist = histogram[i]
        prev_hist = histogram[i - 1] if i > 0 else 0
        
        # ATR filter - avoid trading when volatility is extreme
        atr_pct = atr[i] / price if price > 0 else 0
        if atr_pct > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Check trailing stop for existing positions
        if position_side != 0:
            if position_side == 1:  # Long position
                # Update highest price since entry
                highest_since_entry = max(highest_since_entry, price)
                stoploss_price = highest_since_entry - ATR_STOP_MULT * atr[i]
                
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                    
            elif position_side == -1:  # Short position
                # Update lowest price since entry
                lowest_since_entry = min(lowest_since_entry, price)
                stoploss_price = lowest_since_entry + ATR_STOP_MULT * atr[i]
                
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
        
        # Determine trend direction from KAMA
        kama_trend = 0
        if price > kama[i]:
            kama_trend = 1  # Bullish
        elif price < kama[i]:
            kama_trend = -1  # Bearish
        
        # MACD histogram cross for entry timing
        macd_bullish_cross = (prev_hist <= 0 and current_hist > 0)
        macd_bearish_cross = (prev_hist >= 0 and current_hist < 0)
        
        # ADX trend strength
        trend_strong = current_adx > ADX_STRONG
        trend_moderate = current_adx > ADX_MODERATE
        
        # Generate signals
        if kama_trend == 1 and trend_strong:  # Strong uptrend
            if macd_bullish_cross:
                # Fresh long entry
                signals[i] = SIZE_FULL
                position_side = 1
                highest_since_entry = price
                entry_atr = atr[i]
            elif position_side == 1 and trend_moderate:
                # Hold long position
                signals[i] = SIZE_HALF if current_adx < ADX_STRONG else SIZE_FULL
            else:
                # Exit if trend weakens
                if position_side == 1 and not trend_moderate:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                else:
                    signals[i] = signals[i - 1] if i > 0 else 0.0
                    
        elif kama_trend == -1 and trend_strong:  # Strong downtrend
            if macd_bearish_cross:
                # Fresh short entry
                signals[i] = -SIZE_FULL
                position_side = -1
                lowest_since_entry = price
                entry_atr = atr[i]
            elif position_side == -1 and trend_moderate:
                # Hold short position
                signals[i] = -SIZE_HALF if current_adx < ADX_STRONG else -SIZE_FULL
            else:
                # Exit if trend weakens
                if position_side == -1 and not trend_moderate:
                    signals[i] = 0.0
                    position_side = 0
                    lowest_since_entry = 0.0
                else:
                    signals[i] = signals[i - 1] if i > 0 else 0.0
        else:
            # No clear trend or weak trend
            if position_side != 0 and not trend_moderate:
                signals[i] = 0.0
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = signals[i - 1] if i > 0 else 0.0
    
    return signals