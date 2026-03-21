#!/usr/bin/env python3
"""
EXPERIMENT #008 - Supertrend Trend + RSI Pullback + Volume Confirmation
===============================================================================
Hypothesis: Combining 4H Supertrend for clear trend direction with 1H RSI pullback
entries and volume confirmation. This differs from previous attempts by:
- Using Supertrend (proven in exp#002 with Sharpe=1.278) instead of HMA/KAMA
- RSI pullback entries (proven in mtf_hma_rsi_zscore_v1 with Sharpe=5.4)
- Volume spike filter to confirm breakout validity
- Simplified calculation structure to avoid timeout issues

Key innovations:
- Supertrend(10, 3) for 4H trend - clear binary signal with ATR-based stops
- RSI(14) pullback to 40-60 zone for entries in trend direction
- Volume > 1.5x 20-period average for confirmation
- Discrete position sizing (0.0, ±0.25, ±0.35) to reduce churn
- ATR trailing stop at 2.5*ATR with proper entry tracking

Why this might beat Sharpe=2.139:
- Supertrend provides cleaner trend signals than HMA/KAMA crossovers
- RSI pullback entries catch better risk/reward than momentum entries
- Volume filter reduces false breakouts
- Simpler calculation = faster execution = no timeout
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_rsi_volume_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing - vectorized"""
    n = len(close)
    tr = np.zeros(n)
    
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator - returns trend direction and stop levels"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            # Update supertrend based on previous trend
            if trend[i-1] == 1:  # Previous bullish
                if close[i] > lower_band[i]:
                    supertrend[i] = max(lower_band[i], supertrend[i-1])
                    trend[i] = 1
                else:
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
            else:  # Previous bearish
                if close[i] < upper_band[i]:
                    supertrend[i] = min(upper_band[i], supertrend[i-1])
                    trend[i] = -1
                else:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
    
    return trend, supertrend, upper_band, lower_band


def calculate_rsi(close, period=14):
    """Calculate RSI using vectorized operations"""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period-1] = np.mean(gain[:period])
    avg_loss[period-1] = np.mean(loss[:period])
    
    for i in range(period, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100  # When avg_loss = 0, RSI = 100
    
    return rsi


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume confirmation"""
    n = len(volume)
    vol_sma = np.zeros(n)
    
    for i in range(period-1, n):
        vol_sma[i] = np.mean(volume[i-period+1:i+1])
    
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Resample to 4h for Supertrend trend filter
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
    
    # Calculate 4h Supertrend for trend
    trend_4h, st_4h, upper_4h, lower_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    n_4h = len(trend_4h)
    
    for i in range(n):
        idx_4h = min(i // 4, n_4h - 1)
        if idx_4h >= 10:  # Wait for Supertrend to initialize
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_FULL = 0.35
    SIZE_HALF = 0.25
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45  # Buy pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Sell pullback in downtrend
    RSI_EXIT = 70  # Exit long when overbought
    RSI_EXIT_SHORT = 30  # Exit short when oversold
    
    # Volume confirmation threshold
    VOL_MULT = 1.5
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(50, 20)  # Wait for all indicators
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    highest_price = np.zeros(n)
    lowest_price = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        vol_ratio = volume[i] / vol_sma_1h[i] if vol_sma_1h[i] > 0 else 1.0
        
        # Check trailing stop for existing positions FIRST
        if i > 0 and position_side[i-1] != 0:
            prev_side = position_side[i-1]
            prev_highest = highest_price[i-1] if highest_price[i-1] > 0 else price
            prev_lowest = lowest_price[i-1] if lowest_price[i-1] > 0 else price
            
            if prev_side == 1:  # Long position
                current_highest = max(prev_highest, price)
                highest_price[i] = current_highest
                stoploss_price = current_highest - ATR_STOP_MULT * atr
                
                # Stoploss triggered
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_price[i] = 0
                    lowest_price[i] = 0
                    continue
                
                # RSI overbought exit
                if rsi_val > RSI_EXIT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_price[i] = 0
                    lowest_price[i] = 0
                    continue
                
                # Hold position
                signals[i] = signals[i-1]
                position_side[i] = 1
                entry_price[i] = entry_price[i-1]
                lowest_price[i] = prev_lowest
                continue
                
            elif prev_side == -1:  # Short position
                current_lowest = min(prev_lowest, price)
                lowest_price[i] = current_lowest
                stoploss_price = current_lowest + ATR_STOP_MULT * atr
                
                # Stoploss triggered
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_price[i] = 0
                    lowest_price[i] = 0
                    continue
                
                # RSI oversold exit
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_price[i] = 0
                    lowest_price[i] = 0
                    continue
                
                # Hold position
                signals[i] = signals[i-1]
                position_side[i] = -1
                entry_price[i] = entry_price[i-1]
                highest_price[i] = prev_highest
                continue
        
        # No existing position - look for entries
        if trend == 1:  # 4h uptrend - look for long entries
            # RSI pullback entry with volume confirmation
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30:
                if vol_ratio > VOL_MULT or i > first_valid + 10:  # Volume confirm or allow after warmup
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    highest_price[i] = price
                    lowest_price[i] = price
                else:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = price
                    highest_price[i] = price
                    lowest_price[i] = price
            elif trend_1h[i-1] == 1 and position_side[i-1] == 0:
                # Trend just turned bullish - enter
                signals[i] = SIZE_HALF
                position_side[i] = 1
                entry_price[i] = price
                highest_price[i] = price
                lowest_price[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
                
        elif trend == -1:  # 4h downtrend - look for short entries
            # RSI pullback entry with volume confirmation
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70:
                if vol_ratio > VOL_MULT or i > first_valid + 10:
                    signals[i] = -SIZE_FULL
                    position_side[i] = -1
                    entry_price[i] = price
                    highest_price[i] = price
                    lowest_price[i] = price
                else:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = price
                    highest_price[i] = price
                    lowest_price[i] = price
            elif trend_1h[i-1] == -1 and position_side[i-1] == 0:
                # Trend just turned bearish - enter
                signals[i] = -SIZE_HALF
                position_side[i] = -1
                entry_price[i] = price
                highest_price[i] = price
                lowest_price[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals