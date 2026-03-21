#!/usr/bin/env python3
"""
EXPERIMENT #018 - EMA Trend + Stochastic Entry + ADX Filter + ATR Stop
===============================================================================
Hypothesis: EMA(21/55) crossover on 4h provides cleaner trend signals than Donchian.
1h Stochastic (14,3,3) offers better entry timing than RSI in trending markets.
ADX(14) > 25 filters out choppy regimes where trend strategies fail.
ATR(14) trailing stop at 2.5x protects capital during reversals.

Key innovations vs mtf_donchian_hma_rsi_zscore_v1:
- EMA crossover instead of Donchian (smoother, less whipsaw)
- Stochastic instead of RSI (better for trending entries)
- ADX regime filter (avoid trading in low-trend environments)
- Simpler logic = fewer signal changes = lower fees

Why this might beat Sharpe=2.139:
- EMA crossovers proven in traditional trend following
- Stochastic captures momentum shifts earlier than RSI
- ADX filter prevents losses in choppy markets (major DD source)
- Multi-timeframe structure proven in best strategies
"""

import numpy as np
import pandas as pd

name = "mtf_ema_stoch_adx_atr_v1"
timeframe = "1h"
leverage = 1.0


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    n = len(close)
    ema = np.zeros(n)
    multiplier = 2 / (period + 1)
    
    # Start with SMA for first value
    if n >= period:
        ema[period - 1] = np.mean(close[:period])
        for i in range(period, n):
            ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator (%K and %D)"""
    n = len(close)
    k = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        highest_high = np.max(high[i - k_period + 1:i + 1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 0:
            k[i] = 100 * (close[i] - lowest_low) / range_hl
        else:
            k[i] = 50
    
    # %D is SMA of %K
    if n >= d_period:
        for i in range(k_period - 1 + d_period - 1, n):
            d[i] = np.mean(k[i - d_period + 1:i + 1])
    
    return k, d


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
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
    
    # Smooth with Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initialize at period
    if n >= period * 2:
        plus_di[period * 2 - 1] = 100 * np.sum(plus_dm[1:period * 2]) / np.sum(tr[1:period * 2]) if np.sum(tr[1:period * 2]) > 0 else 0
        minus_di[period * 2 - 1] = 100 * np.sum(minus_dm[1:period * 2]) / np.sum(tr[1:period * 2]) if np.sum(tr[1:period * 2]) > 0 else 0
        
        if plus_di[period * 2 - 1] + minus_di[period * 2 - 1] > 0:
            dx[period * 2 - 1] = 100 * abs(plus_di[period * 2 - 1] - minus_di[period * 2 - 1]) / (plus_di[period * 2 - 1] + minus_di[period * 2 - 1])
        
        adx[period * 2 - 1] = dx[period * 2 - 1]
        
        # Smooth ADX
        for i in range(period * 2, n):
            plus_di[i] = (plus_di[i - 1] * (period - 1) + 100 * plus_dm[i] / tr[i] if tr[i] > 0 else plus_di[i - 1] * (period - 1)) / period
            minus_di[i] = (minus_di[i - 1] * (period - 1) + 100 * minus_dm[i] / tr[i] if tr[i] > 0 else minus_di[i - 1] * (period - 1)) / period
            
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


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
    if n >= period:
        atr[period - 1] = np.mean(tr[1:period])
        
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    atr_1h = calculate_atr(high, low, close, period=14)
    stoch_k_1h, stoch_d_1h = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    
    # Resample to 4h for trend and regime filters
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # 4h EMA crossover for trend
    ema21_4h = calculate_ema(c_4h, 21)
    ema55_4h = calculate_ema(c_4h, 55)
    
    # 4h ADX for regime filter
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Determine 4h trend direction
    trend_4h = np.zeros(len(c_4h))
    for i in range(55, len(c_4h)):
        if ema21_4h[i] > ema55_4h[i]:
            trend_4h[i] = 1  # Bullish
        elif ema21_4h[i] < ema55_4h[i]:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h signals back to 1h
    trend_1h = np.zeros(n)
    adx_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
        if idx_4h < len(adx_4h):
            adx_1h[i] = adx_4h[idx_4h]
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_FULL = 0.35
    SIZE_HALF = 0.20
    
    # Thresholds
    ADX_MIN = 25  # Minimum ADX for trending regime
    STOCH_LONG = 30  # Stochastic oversold for long entry
    STOCH_SHORT = 70  # Stochastic overbought for short entry
    
    # ATR stoploss
    ATR_STOP_MULT = 2.5
    
    first_valid = max(55 * 4, 14, 28)  # Wait for all indicators
    
    # Track positions for stoploss
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    highest_price = np.zeros(n)  # For trailing long stops
    lowest_price = np.zeros(n)  # For trailing short stops
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(stoch_k_1h[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        adx_val = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        stoch_k = stoch_k_1h[i]
        stoch_d = stoch_d_1h[i]
        
        # ADX regime filter - only trade in trending markets
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            continue
        
        # ATR volatility filter
        if atr > 0 and atr / price > 0.05:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_highest = highest_price[i - 1] if highest_price[i - 1] > 0 else price
            prev_lowest = lowest_price[i - 1] if lowest_price[i - 1] > 0 else price
            
            # Update extreme prices
            if prev_side == 1:
                current_highest = max(prev_highest, price)
                stoploss_price = current_highest - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_price[i] = 0
                    continue
                highest_price[i] = current_highest
                lowest_price[i] = 0
            elif prev_side == -1:
                current_lowest = min(prev_lowest, price)
                stoploss_price = current_lowest + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    lowest_price[i] = 0
                    continue
                lowest_price[i] = current_lowest
                highest_price[i] = 0
            
            # Hold position if no stop triggered
            if position_side[i - 1] != 0 and position_side[i] == 0:
                pass  # Already closed by stop
            elif position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                continue
        
        # Entry logic
        if trend == 1:  # 4h uptrend
            # Stochastic pullback entry
            if stoch_k < STOCH_LONG and stoch_d < STOCH_LONG + 5:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                highest_price[i] = price
            elif stoch_k < 50 and stoch_d < 50:
                signals[i] = SIZE_HALF
                position_side[i] = 1
                entry_price[i] = price
                highest_price[i] = price
            else:
                # Hold or exit on trend reversal
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # Stochastic rally entry
            if stoch_k > STOCH_SHORT and stoch_d > STOCH_SHORT - 5:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                lowest_price[i] = price
            elif stoch_k > 50 and stoch_d > 50:
                signals[i] = -SIZE_HALF
                position_side[i] = -1
                entry_price[i] = price
                lowest_price[i] = price
            else:
                # Hold or exit on trend reversal
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals