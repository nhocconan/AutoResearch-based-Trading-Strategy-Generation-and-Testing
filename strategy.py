#!/usr/bin/env python3
"""
EXPERIMENT #021 - KAMA + MACD Histogram + Volume + HTF Trend Filter (1h primary, 4h HTF)
========================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better 
than fixed EMAs, reducing whipsaws in choppy markets. Combined with MACD histogram 
momentum shifts, volume confirmation, and 4h HTF trend filter, this should capture 
trending moves while avoiding false breakouts. ADX > 20 ensures we only trade in 
trending regimes.

Key features:
- Primary TF: 1h (required for this experiment)
- HTF filter: 4h KAMA(21) for major trend direction
- Trend: KAMA(10) vs KAMA(40) crossover on 1h
- Momentum: MACD(12,26,9) histogram turning positive/negative
- Volume: volume > 1.5 * 20-period average (confirms breakout)
- Regime: ADX(14) > 20 (trending market only)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit, trail stop

Why this differs from failed attempts:
- KAMA adapts efficiency ratio based on price movement (unlike fixed EMA/HMA)
- MACD histogram (not signal line cross) gives earlier momentum signals
- Volume confirmation reduces false breakouts
- ADX regime filter avoids choppy markets (major failure point in #019, #020)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_macd_volume_adx_1h_4h_v2"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average
    KAMA adapts to market volatility using Efficiency Ratio (ER)
    ER = |net change| / sum of absolute changes over period
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period - 1, n):
        net_change = abs(close[i] - close[i - period + 1])
        sum_changes = 0.0
        for j in range(i - period + 2, i + 1):
            sum_changes += abs(close[j] - close[j - 1])
        if sum_changes > 0:
            er[i] = net_change / sum_changes
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(period - 1, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period - 1] = close[period - 1]
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
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
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period - 1, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX
    dx = np.zeros(n)
    for i in range(period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    kama_4h = calculate_kama(df_4h['close'].values, 21)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 1h indicators
    kama_fast = calculate_kama(close, 10, 2, 30)
    kama_slow = calculate_kama(close, 40, 2, 30)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    adx = calculate_adx(high, low, close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    # Volume moving average
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(kama_fast[i]) or 
            np.isnan(kama_slow[i]) or np.isnan(macd_hist[i]) or 
            np.isnan(adx[i]) or np.isnan(atr[i]) or np.isnan(volume_sma[i]) or 
            atr[i] == 0 or volume_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HTF trend filter
        htf_trend = 1 if close[i] > kama_4h_aligned[i] else -1
        
        # 1h KAMA crossover trend
        kama_trend = 1 if kama_fast[i] > kama_slow[i] else -1
        
        # MACD histogram momentum (turning positive/negative)
        macd_momentum = 0
        if i > 0 and not np.isnan(macd_hist[i - 1]):
            if macd_hist[i] > 0 and macd_hist[i - 1] <= 0:
                macd_momentum = 1  # Bullish momentum shift
            elif macd_hist[i] < 0 and macd_hist[i - 1] >= 0:
                macd_momentum = -1  # Bearish momentum shift
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * volume_sma[i]
        
        # ADX regime filter (trending market only)
        regime_valid = adx[i] > 20
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HTF bullish + KAMA bullish + MACD momentum + Volume + ADX
        if htf_trend == 1 and kama_trend == 1 and macd_momentum == 1 and volume_confirmed and regime_valid:
            target_signal = SIZE
        
        # Short entry: HTF bearish + KAMA bearish + MACD momentum + Volume + ADX
        elif htf_trend == -1 and kama_trend == -1 and macd_momentum == -1 and volume_confirmed and regime_valid:
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
                if position_side == 1 and kama_trend == -1:
                    # KAMA trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and kama_trend == 1:
                    # KAMA trend reversed, exit short
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