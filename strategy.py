#!/usr/bin/env python3
"""
EXPERIMENT #094 - KAMA_ADX_MTF_TREND_PULLBACK_15M_1H_4H_V1
==================================================================================================
Hypothesis: Recent ensemble failures (#086, #090, #092) had massive drawdowns due to:
1. Complex voting logic creating whipsaws and excessive churn (fees kill returns)
2. Position tracking bugs in stoploss/TP logic
3. Not filtering for trend strength (trading in choppy markets)

Key changes from #093:
- SIMPLER signal logic: KAMA trend + ADX strength filter + RSI pullback (not 3-way voting)
- KAMA adapts to volatility better than HMA (less whipsaw in chop)
- ADX > 25 filter: only trade when trend is strong (skip choppy regimes)
- Wider stoploss: 2.5*ATR (avoid premature exits from noise)
- Higher TP: 3R (let winners run, improve win rate)
- More conservative sizing: 0.25 max (not 0.35)
- Cleaner position tracking (no state bugs)

Why this should beat #093 (Sharpe=0.189):
- ADX filter avoids trading in choppy markets (major source of losses)
- KAMA smoother than HMA = fewer false signals
- Simpler logic = less churn = lower fees
- Based on winning pattern from #084 (Sharpe=0.423) which used KAMA+Supertrend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_adx_mtf_trend_pullback_15m_1h_4h_v1"
timeframe = "15m"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)"""
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth using Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initialize first period
    sum_tr = np.sum(tr[1:period + 1])
    sum_plus_dm = np.sum(plus_dm[1:period + 1])
    sum_minus_dm = np.sum(minus_dm[1:period + 1])
    
    for i in range(period, n):
        if i == period:
            tr_smooth = sum_tr
            plus_dm_smooth = sum_plus_dm
            minus_dm_smooth = sum_minus_dm
        else:
            tr_smooth = (tr_smooth * (period - 1) + tr[i]) / period
            plus_dm_smooth = (plus_dm_smooth * (period - 1) + plus_dm[i]) / period
            minus_dm_smooth = (minus_dm_smooth * (period - 1) + minus_dm[i]) / period
        
        if tr_smooth > 0:
            plus_di[i] = 100 * plus_dm_smooth / tr_smooth
            minus_di[i] = 100 * minus_dm_smooth / tr_smooth
        else:
            plus_di[i] = 0
            minus_di[i] = 0
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # Smooth DX to get ADX
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # Get 1h HTF data using mtf_data helper (PROPER alignment)
    try:
        df_1h = get_htf_data(prices, '1h')
        close_1h = df_1h['close'].values
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        
        # 1h KAMA for trend (smoother than HMA, adapts to volatility)
        kama_1h_raw = calculate_kama(close_1h, period=10, fast=2, slow=30)
        
        # 1h ADX for trend strength filter
        adx_1h_raw = calculate_adx(high_1h, low_1h, close_1h, period=14)
        
        # Align HTF indicators to LTF (auto shift(1) for completed bars)
        kama_1h = align_htf_to_ltf(prices, df_1h, kama_1h_raw)
        adx_1h = align_htf_to_ltf(prices, df_1h, adx_1h_raw)
    except Exception:
        kama_1h = np.zeros(n)
        adx_1h = np.zeros(n)
    
    # Get 4h HTF data for additional trend confirmation
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h KAMA for major trend
        kama_4h_raw = calculate_kama(close_4h, period=10, fast=2, slow=30)
        
        # Align to 15m
        kama_4h = align_htf_to_ltf(prices, df_4h, kama_4h_raw)
    except Exception:
        kama_4h = np.zeros(n)
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_FULL = 0.25  # Max position (25% of capital)
    SIZE_HALF = 0.125  # Half position after TP
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ATR stoploss/take profit multipliers
    ATR_STOP_MULT = 2.5  # Wider stop to avoid noise
    ATR_TP_MULT = 3.0  # Let winners run (3R)
    
    # ADX threshold for trend strength
    ADX_MIN = 25  # Only trade when trend is strong
    
    first_valid = max(200, 14 * 3, 30)
    
    # Track position state
    in_position = np.zeros(n, dtype=bool)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Validate data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] <= 0:
            signals[i] = 0.0
            in_position[i] = False
            position_side[i] = 0
            continue
        
        # Get 1h trend from KAMA
        kama_trend_1h = 0
        if kama_1h[i] > 0 and not np.isnan(kama_1h[i]):
            if close[i] > kama_1h[i]:
                kama_trend_1h = 1
            elif close[i] < kama_1h[i]:
                kama_trend_1h = -1
        
        # Get 4h trend from KAMA
        kama_trend_4h = 0
        if kama_4h[i] > 0 and not np.isnan(kama_4h[i]):
            if close[i] > kama_4h[i]:
                kama_trend_4h = 1
            elif close[i] < kama_4h[i]:
                kama_trend_4h = -1
        
        # Get ADX for trend strength
        adx_val = adx_1h[i] if adx_1h[i] > 0 and not np.isnan(adx_1h[i]) else 0
        
        # Get 15m Supertrend for entry timing
        st_trend_15m = st_direction_15m[i]
        rsi_15m_val = rsi_15m[i]
        
        # Check existing position for exits
        if in_position[i - 1]:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1]
            prev_tp = tp_triggered[i - 1]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
                highest_since_entry[i] = max(prev_high, high[i])
                lowest_since_entry[i] = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else low[i]
            else:
                prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
                highest_since_entry[i] = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else high[i]
                lowest_since_entry[i] = min(prev_low, low[i])
            
            current_high = highest_since_entry[i]
            current_low = lowest_since_entry[i]
            
            # Stoploss check (2.5*ATR)
            exit_signal = False
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if low[i] < stoploss_price:
                    exit_signal = True
                
                # Take profit check (3R) - reduce to half
                if not prev_tp:
                    tp_price = prev_entry + ATR_TP_MULT * atr_15m[i]
                    if high[i] >= tp_price:
                        signals[i] = SIZE_HALF
                        in_position[i] = True
                        position_side[i] = 1
                        entry_price[i] = prev_entry
                        tp_triggered[i] = True
                        highest_since_entry[i] = current_high
                        lowest_since_entry[i] = current_low
                        continue
                
                # Trail stop at 1.5R profit after TP triggered
                if prev_tp:
                    trail_stop = current_high - 1.5 * atr_15m[i]
                    if low[i] < trail_stop:
                        exit_signal = True
            else:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if high[i] > stoploss_price:
                    exit_signal = True
                
                # Take profit check (3R) - reduce to half
                if not prev_tp:
                    tp_price = prev_entry - ATR_TP_MULT * atr_15m[i]
                    if low[i] <= tp_price:
                        signals[i] = -SIZE_HALF
                        in_position[i] = True
                        position_side[i] = -1
                        entry_price[i] = prev_entry
                        tp_triggered[i] = True
                        highest_since_entry[i] = current_high
                        lowest_since_entry[i] = current_low
                        continue
                
                # Trail stop at 1.5R profit after TP triggered
                if prev_tp:
                    trail_stop = current_low + 1.5 * atr_15m[i]
                    if high[i] > trail_stop:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                in_position[i] = False
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                # Hold position
                signals[i] = signals[i - 1]
                in_position[i] = in_position[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
            continue
        
        # ENTRY LOGIC: Need ALL conditions to agree
        # 1. 1h KAMA trend direction
        # 2. 4h KAMA confirms (same direction)
        # 3. ADX > 25 (strong trend, not choppy)
        # 4. 15m RSI pullback in trend direction
        # 5. 15m Supertrend confirms
        
        # Check ADX filter first (skip if trend is weak)
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            in_position[i] = False
            position_side[i] = 0
            continue
        
        # LONG entry conditions
        long_condition = (
            kama_trend_1h == 1 and  # 1h trend up
            kama_trend_4h >= 0 and  # 4h trend up or neutral
            RSI_LONG_MIN <= rsi_15m_val <= RSI_LONG_MAX and  # RSI pullback
            st_trend_15m == 1  # Supertrend confirms
        )
        
        # SHORT entry conditions
        short_condition = (
            kama_trend_1h == -1 and  # 1h trend down
            kama_trend_4h <= 0 and  # 4h trend down or neutral
            RSI_SHORT_MIN <= rsi_15m_val <= RSI_SHORT_MAX and  # RSI pullback
            st_trend_15m == -1  # Supertrend confirms
        )
        
        if long_condition:
            signals[i] = SIZE_FULL
            in_position[i] = True
            position_side[i] = 1
            entry_price[i] = close[i]
            tp_triggered[i] = False
            highest_since_entry[i] = high[i]
            lowest_since_entry[i] = low[i]
        elif short_condition:
            signals[i] = -SIZE_FULL
            in_position[i] = True
            position_side[i] = -1
            entry_price[i] = close[i]
            tp_triggered[i] = False
            highest_since_entry[i] = high[i]
            lowest_since_entry[i] = low[i]
        else:
            signals[i] = 0.0
            in_position[i] = False
            position_side[i] = 0
    
    return signals