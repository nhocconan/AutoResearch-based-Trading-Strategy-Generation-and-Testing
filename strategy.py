#!/usr/bin/env python3
"""
EXPERIMENT #049 - MTF HMA+RSI+ADX+ATR 1h+4h Clean v1
==================================================================================================
Hypothesis: Experiment #048 crashed due to variable scoping issues in stoploss logic.
Current best (mtf_supertrend_macd_bbw_rsi_15m_1h_4h_v1) has Sharpe=3.653.

Key changes from #048:
- Switch to 1h entries + 4h trend (more stable than 15m, less churn)
- Simplified indicator set: HMA + RSI + ADX + ATR (remove Supertrend complexity)
- Fixed variable scoping: all variables defined before use in loop
- Cleaner position tracking with explicit state management
- Discrete signal levels: 0.0, ±0.20, ±0.35 (minimize fee churn)
- Volatility-based position sizing with proper clamping
- Stoploss: 2.5*ATR, Take profit: 2R with trail at 1R

Why this should beat current best:
- 1h timeframe has proven stability (less noise than 15m)
- Simpler logic = fewer bugs (learned from #048 crash)
- HMA is faster than EMA for trend detection
- ADX filter ensures we only trade strong trends
- Proper MTF alignment using mtf_data helper
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_adx_atr_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        raw_vals = 2 * wma1[start_idx:i + 1] - wma2[start_idx:i + 1]
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
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
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    adx_1h = calculate_adx(high, low, close, period=14)
    
    # 4h trend filter using mtf_data helper (CRITICAL - proper alignment)
    df_4h = get_htf_data(prices, '4h')
    
    hma_4h_aligned = np.zeros(n)
    adx_4h_aligned = np.zeros(n)
    
    if df_4h is not None and len(df_4h) >= 50:
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        hma_4h = calculate_hma(c_4h, period=21)
        adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    BASE_SIZE = 0.35
    SIZE_HALF = 0.175
    TARGET_VOL = 0.02  # Target 2% daily volatility
    
    # Entry thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 60
    ADX_MIN_4H = 25  # 4h trend strength
    ADX_MIN_1H = 20  # 1h momentum
    
    # Risk management
    ATR_STOP_MULT = 2.5
    TP_MULT = 2.0  # 2R take profit
    
    first_valid = max(200, 14 * 2, 28)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for NaN or zero ATR
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Get 4h trend signals
        hma_4h_val = hma_4h_aligned[i]
        adx_4h_val = adx_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if hma_4h_val > 0 and close[i] > hma_4h_val:
            trend_4h = 1
        elif hma_4h_val > 0 and close[i] < hma_4h_val:
            trend_4h = -1
        
        # ADX filter (4h) - only trade when trend is strong
        if adx_4h_val < ADX_MIN_4H:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            atr = atr_1h[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, close[i])
                current_low = min(prev_low, close[i]) if prev_low > 0 else close[i]
            else:
                current_high = max(prev_high, close[i]) if prev_high > 0 else close[i]
                current_low = min(prev_low, close[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and close[i] >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and close[i] <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if close[i] > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Volatility-based position sizing
        atr_pct = atr_1h[i] / close[i] if close[i] > 0 else 0.02
        vol_scalar = min(1.5, max(0.5, TARGET_VOL / atr_pct)) if atr_pct > 0 else 1.0
        position_size = BASE_SIZE * vol_scalar
        position_size = min(0.40, max(0.20, position_size))  # Clamp to safe range
        
        # Check 1h ADX for momentum confirmation
        adx_1h_val = adx_1h[i]
        
        # Entry logic: 4h trend + 1h RSI pullback + 1h ADX confirmation
        rsi_val = rsi_1h[i]
        
        if trend_4h == 1 and adx_1h_val >= ADX_MIN_1H:  # Bullish trend confirmed
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:  # Pullback entry
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = False
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
                
        elif trend_4h == -1 and adx_1h_val >= ADX_MIN_1H:  # Bearish trend confirmed
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:  # Pullback entry
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = False
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals