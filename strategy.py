#!/usr/bin/env python3
"""
EXPERIMENT #013 - KAMA Adaptive Trend + MACD Momentum + Volume + HTF Filter (15m primary, 4h HTF)
=================================================================================================
Hypothesis: KAMA adapts to volatility better than EMA/SMA, reducing whipsaws in chop.
MACD histogram confirms momentum direction. Volume spike confirms breakout validity.
4h KAMA provides major trend filter to avoid counter-trend trades on 15m.
ADX(14) > 25 ensures we only trade in trending markets (avoid chop).

Key features:
- Primary TF: 15m (as required for this experiment)
- HTF filter: 4h KAMA(21) for major trend direction
- Trend: KAMA(21) vs KAMA(42) crossover on 15m
- Momentum: MACD(12,26,9) histogram confirmation
- Volume: Volume > 1.5 * 20-bar MA volume (confirms breakout)
- Regime: ADX(14) > 25 (trending market only)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why different from failed strategies:
- KAMA adapts to volatility (unlike fixed EMA/SMA in failed attempts)
- Volume confirmation reduces false breakouts
- ADX regime filter avoids choppy markets (missing in many failed strats)
- 4h HTF filter (not 1d which may be too slow for 15m entries)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_macd_volume_adx_15m_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - faster in trends, slower in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period - 1, n):
        change = abs(close[i] - close[i - period + 1])
        volatility = np.sum(np.abs(np.diff(close[i - period + 1:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    # Calculate KAMA
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] ** 2 * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    if n < period * 2:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's method
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr + 1e-10) * 100
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100
        else:
            dx[i] = 0
    
    # Smooth DX to get ADX
    adx_series = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean()
    adx[:] = adx_series.values
    
    return adx


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    kama_4h_fast = calculate_kama(df_4h['close'].values, 21)
    kama_4h_slow = calculate_kama(df_4h['close'].values, 42)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_fast - kama_4h_slow)  # KAMA spread
    
    # Calculate 15m indicators
    kama_15m_fast = calculate_kama(close, 21)
    kama_15m_slow = calculate_kama(close, 42)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    adx = calculate_adx(high, low, close, 14)
    atr = calculate_atr(high, low, close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(kama_15m_fast[i]) or np.isnan(kama_15m_slow[i]) or
            np.isnan(macd_hist[i]) or np.isnan(adx[i]) or np.isnan(atr[i]) or
            np.isnan(kama_4h_aligned[i]) or np.isnan(vol_ma[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HTF trend filter (KAMA spread)
        htf_trend = 1 if kama_4h_aligned[i] > 0 else -1
        
        # 15m KAMA crossover
        kama_cross = 1 if kama_15m_fast[i] > kama_15m_slow[i] else -1
        
        # MACD histogram momentum confirmation
        macd_momentum = 1 if macd_hist[i] > 0 else -1
        
        # Volume confirmation (volume > 1.5 * 20-bar MA)
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # ADX regime filter (ADX > 25 = trending market)
        regime_valid = adx[i] > 25
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: All filters aligned bullish + volume confirmation + regime valid
        if (kama_cross == 1 and macd_momentum == 1 and htf_trend == 1 and 
            regime_valid and volume_confirmed):
            target_signal = SIZE
        
        # Short entry: All filters aligned bearish + volume confirmation + regime valid
        elif (kama_cross == -1 and macd_momentum == -1 and htf_trend == -1 and 
              regime_valid and volume_confirmed):
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and kama_cross == -1:
                    # KAMA crossed bearish, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and kama_cross == 1:
                    # KAMA crossed bullish, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals