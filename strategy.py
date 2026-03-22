#!/usr/bin/env python3
"""
Experiment #159: 1h EMA Crossover + 4h HMA Trend + Volume Confirmation + ATR Stop

Hypothesis: 1h timeframe provides a balance between noise (15m) and lag (4h).
Simple EMA crossover generates adequate trades, while 4h HMA filters counter-trend
entries. Volume confirmation reduces false breakouts. ATR stoploss protects capital.

Why this might work where others failed:
- #147, #152 (KAMA): Too complex, laggy adaptive MA hurt performance
- #148, #153 (Donchian): Breakout strategies fail in choppy 2022-2024 markets
- #149, #154, #155 (RSI mean rev): Mean reversion fails in strong trends
- #151, #156, #157 (Regime adaptive): Over-engineered, too many filters = 0 trades

This strategy uses PROVEN components:
- 4h HMA trend bias (from baseline that achieved Sharpe=0.478)
- Simple EMA crossover (generates trades, unlike complex filters)
- Volume spike confirmation (filters false breakouts)
- ATR trailing stop (mandatory risk management)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_ema_4h_hma_volume_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average for volume spike detection."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    vol_ma = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === EMA CROSSOVER SIGNAL ===
        # Fast EMA crosses above slow EMA = bullish
        # Fast EMA crosses below slow EMA = bearish
        ema_bull = ema_fast[i] > ema_slow[i]
        ema_bear = ema_fast[i] < ema_slow[i]
        
        # Check for actual crossover (not just position)
        ema_cross_bull = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_bear = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # === VOLUME CONFIRMATION ===
        # Volume must be above average to confirm breakout
        vol_spike = volume[i] > 1.2 * vol_ma[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # 4h bullish + EMA bullish + volume confirmation
        # Allow entries on crossover OR sustained bullish alignment
        if bull_trend_4h and ema_bull:
            if ema_cross_bull and vol_spike:
                new_signal = SIZE_STRONG  # Strong signal on crossover + volume
            elif ema_bull:
                new_signal = SIZE_BASE  # Maintain position on sustained trend
        
        # === SHORT ENTRY CONDITIONS ===
        # 4h bearish + EMA bearish + volume confirmation
        if bear_trend_4h and ema_bear:
            if ema_cross_bear and vol_spike:
                new_signal = -SIZE_STRONG  # Strong signal on crossover + volume
            elif ema_bear:
                new_signal = -SIZE_BASE  # Maintain position on sustained trend
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals