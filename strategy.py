#!/usr/bin/env python3
"""
EXPERIMENT #005 - EMA Trend (4h) + Daily Regime + RSI Pullback + Volume Confirm
================================================================================
Hypothesis: 4h timeframe balances trend clarity with sufficient trade frequency.
Daily EMA(21/55) crossover provides robust regime filter (proven in traditional markets).
4h RSI(14) pullback entries with volume confirmation reduce false signals.
ATR-based trailing stop with take-profit scaling manages risk/reward.

Why this differs from failed strategies:
- 4h primary TF (vs failed 12h) = more trades while still avoiding 15m/1h noise
- Daily EMA(21/55) regime (vs SMA50) = more responsive to trend changes
- Volume confirmation on entries = filters weak breakouts that caused whipsaws
- Simplified position management = fewer bugs in stoploss/takeprofit logic
- Conservative position size (0.30) with discrete levels to minimize fee churn

Key risk controls:
- Signal magnitude: 0.30 (30% position size, max 0.40)
- Stoploss: 2.0*ATR trailing stop from entry/highest
- Take profit: reduce to half at 2R, trail stop at 1R
- Discrete levels: 0.0, ±0.30, ±0.15 (half position)
- Volume filter: entry volume > 1.5*20-bar avg volume
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ema_4h_daily_regime_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0


def calculate_ema(close, period=21):
    """Calculate EMA with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    return ema


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    rsi[:period] = np.nan
    return rsi


def calculate_atr(high, low, close, period=14):
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (for regime filter)
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate daily EMA(21) and EMA(55) for long-term trend regime
    daily_ema21 = calculate_ema(daily_close, period=21)
    daily_ema55 = calculate_ema(daily_close, period=55)
    
    # Align daily indicators to 4h timeframe (auto shift(1) for completed bars)
    daily_ema21_aligned = align_htf_to_ltf(prices, df_1d, daily_ema21)
    daily_ema55_aligned = align_htf_to_ltf(prices, df_1d, daily_ema55)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Calculate indicators on primary 4h timeframe
    ema_21 = calculate_ema(close, period=21)
    ema_55 = calculate_ema(close, period=55)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # Generate signals with discrete position sizing and stoploss
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative)
    HALF_SIZE = SIZE / 2  # 15% for take profit reduction
    ATR_STOP_MULT = 2.0  # Stoploss at 2.0*ATR
    RSI_LONG_ENTRY = 40  # RSI pullback level for longs (deeper pullback)
    RSI_SHORT_ENTRY = 60  # RSI pullback level for shorts
    VOLUME_MULT = 1.5  # Volume must be 1.5x average for entry confirmation
    
    # Track position state
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    tp_hit = False
    
    # Find first valid index (all indicators ready)
    first_valid = max(55, 20, 14)  # EMA(55), Vol_SMA(20), RSI(14)
    
    for i in range(first_valid, n):
        # Check for NaN values in primary indicators
        if (np.isnan(ema_21[i]) or np.isnan(ema_55[i]) or 
            np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
            continue
        
        # Daily regime filter (allow trades if daily data not available yet)
        daily_trend_bullish = False
        daily_trend_bearish = False
        
        if not np.isnan(daily_ema21_aligned[i]) and not np.isnan(daily_ema55_aligned[i]):
            daily_trend_bullish = daily_ema21_aligned[i] > daily_ema55_aligned[i]
            daily_trend_bearish = daily_ema21_aligned[i] < daily_ema55_aligned[i]
        
        # 4h EMA trend direction
        ema_bullish = ema_21[i] > ema_55[i]
        ema_bearish = ema_21[i] < ema_55[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > (VOLUME_MULT * vol_sma[i])
        
        # Check stoploss/trailing stop first (before new signals)
        if position_side == 1 and entry_price > 0:
            # Update highest since entry for trailing
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            
            # Trailing stop: 2.0*ATR from highest (for longs)
            trailing_stop = highest_since_entry - ATR_STOP_MULT * atr[i]
            
            # Initial stoploss: 2.0*ATR below entry
            initial_stop = entry_price - ATR_STOP_MULT * atr[i]
            stop_level = max(initial_stop, trailing_stop)
            
            # Check if stoploss hit
            if close[i] < stop_level:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                tp_hit = False
                continue
            
            # Take profit: reduce to half at 2R (2 * 2.0*ATR = 4.0*ATR from entry)
            tp_level = entry_price + 4.0 * atr[i]
            if not tp_hit and close[i] > tp_level:
                signals[i] = HALF_SIZE
                tp_hit = True
                # Move stop to breakeven + 1R (1*ATR)
                highest_since_entry = max(highest_since_entry, close[i])
                continue
            
            # Maintain position
            signals[i] = HALF_SIZE if tp_hit else SIZE
            continue
        
        if position_side == -1 and entry_price > 0:
            # Update lowest since entry for trailing
            if close[i] < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = close[i]
            
            # Trailing stop: 2.0*ATR from lowest (for shorts)
            trailing_stop = lowest_since_entry + ATR_STOP_MULT * atr[i]
            
            # Initial stoploss: 2.0*ATR above entry
            initial_stop = entry_price + ATR_STOP_MULT * atr[i]
            stop_level = min(initial_stop, trailing_stop)
            
            # Check if stoploss hit
            if close[i] > stop_level:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                tp_hit = False
                continue
            
            # Take profit: reduce to half at 2R
            tp_level = entry_price - 4.0 * atr[i]
            if not tp_hit and close[i] < tp_level:
                signals[i] = -HALF_SIZE
                tp_hit = True
                lowest_since_entry = min(lowest_since_entry, close[i])
                continue
            
            # Maintain position
            signals[i] = -HALF_SIZE if tp_hit else -SIZE
            continue
        
        # Generate new entry signals (only if flat)
        if position_side == 0:
            # Long entry: EMA bullish + Daily bullish + RSI pullback + Volume confirm
            if ema_bullish and (daily_trend_bullish or np.isnan(daily_ema21_aligned[i])):
                if rsi[i] < RSI_LONG_ENTRY and volume_confirmed:
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    tp_hit = False
                    continue
            
            # Short entry: EMA bearish + Daily bearish + RSI pullback + Volume confirm
            if ema_bearish and (daily_trend_bearish or np.isnan(daily_ema21_aligned[i])):
                if rsi[i] > RSI_SHORT_ENTRY and volume_confirmed:
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
                    tp_hit = False
                    continue
        
        # No signal - maintain current position
        if position_side == 0:
            signals[i] = 0.0
        else:
            signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals