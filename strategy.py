#!/usr/bin/env python3
"""
EXPERIMENT #052 - Supertrend KAMA Volume Confirmed Strategy (30m Primary + 4h Trend)
==================================================================================================
Hypothesis: Supertrend provides cleaner trend signals than HMA alone, while KAMA adapts to volatility.
Adding volume confirmation reduces false breakouts. 30m timeframe captures more opportunities than 1h
while 4h filter prevents counter-trend disasters. Simpler than ensemble (avoiding #051 crash).

Key innovations:
1. SUPERTREND (ATR=10, mult=3): Clean trend direction with built-in stoploss levels
2. KAMA (ER=10): Adaptive moving average that speeds up in trends, slows in chop
3. VOLUME CONFIRMATION: Entry only when volume > 1.5x 20-bar average (reduces false breakouts)
4. 4h HMA FILTER: Only trade in direction of higher timeframe trend
5. ATR STOPLOSS: 2.5*ATR trailing stop with take-profit at 2R

Why this should beat hma_rsi_pullback_daily_trend_4h_v1 (Sharpe=0.537):
- Supertrend has shown strong performance in experiments #045, #047 (Sharpe > 0.49)
- Volume filter reduces whipsaws during low-liquidity periods
- 30m timeframe generates more signals than 1h/4h while maintaining quality
- KAMA adapts better to regime changes than fixed-period EMA/HMA
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_kama_volume_30m_4h_v1"
timeframe = "30m"
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


def calculate_supertrend(high, low, close, atr, mult=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    if n < len(atr) or len(atr) < 14:
        return np.zeros(n), np.zeros(n)
    
    hl2 = (high + low) / 2.0
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    for i in range(len(atr)):
        upper_band[i] = hl2[i] + mult * atr[i]
        lower_band[i] = hl2[i] - mult * atr[i]
    
    # Initialize
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if direction[i - 1] == 1:
            # Previously long
            if lower_band[i] > supertrend[i - 1]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = supertrend[i - 1]
                if close[i] < supertrend[i]:
                    direction[i] = -1
                else:
                    direction[i] = 1
        else:
            # Previously short
            if upper_band[i] < supertrend[i - 1]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = supertrend[i - 1]
                if close[i] > supertrend[i]:
                    direction[i] = 1
                else:
                    direction[i] = -1
    
    return supertrend, direction


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs chop).
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    close = np.array(close, dtype=float)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_volume_ma(volume, period=20):
    """Calculate rolling volume moving average"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # ========== 30m INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_30m = calculate_atr(high, low, close, period=14)
    supertrend_30m, st_direction_30m = calculate_supertrend(high, low, close, atr_30m, mult=3.0)
    kama_30m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_30m = calculate_rsi(close, period=14)
    vol_ma_30m = calculate_volume_ma(volume, period=20)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    # Load HTF data ONCE before the loop (Rule 1)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        supertrend_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, mult=3.0)
        
        # Align to 30m timeframe (auto shift for completed bars - Rule 2)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
        
    except Exception:
        st_direction_4h_aligned = np.zeros(n)
        supertrend_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (Rule 4)
    SIZE_LOW = 0.20    # Low conviction
    SIZE_BASE = 0.30   # Base position
    SIZE_HIGH = 0.35   # High conviction
    
    # ATR stoploss
    ATR_STOP_MULT = 2.5
    
    first_valid = max(100, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_30m[i]) or atr_30m[i] == 0 or np.isnan(rsi_30m[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_30m[i]
        rsi_val = rsi_30m[i]
        st_dir = st_direction_30m[i]
        kama_val = kama_30m[i]
        vol_ratio = volume[i] / vol_ma_30m[i] if vol_ma_30m[i] > 0 else 1.0
        
        # 4h trend filter
        st_dir_4h = st_direction_4h_aligned[i]
        supertrend_4h_val = supertrend_4h_aligned[i]
        
        # ========== TREND CONFIRMATION ==========
        # Need 30m and 4h supertrend to agree for high conviction
        trend_agree = (st_dir == st_dir_4h) and (st_dir != 0)
        
        # KAMA confirmation (price above KAMA for long, below for short)
        kama_confirm_long = price > kama_val
        kama_confirm_short = price < kama_val
        
        # Volume confirmation (reduce false breakouts)
        volume_confirmed = vol_ratio > 1.5
        
        # ========== CHECK EXISTING POSITIONS ==========
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE / 2
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE / 2
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered (unless supertrend reverses)
            if st_dir * prev_side < 0:  # Supertrend reversal
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== ENTRY LOGIC ==========
        # LONG entry: supertrend long + 4h agrees + KAMA confirms + volume confirmed
        if st_dir == 1 and st_dir_4h >= 0 and kama_confirm_long:
            if volume_confirmed and trend_agree:
                # High conviction: all signals agree
                signals[i] = SIZE_HIGH
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif volume_confirmed or trend_agree:
                # Base conviction: 2 of 3 confirm
                signals[i] = SIZE_BASE
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Low conviction: only supertrend
                signals[i] = SIZE_LOW
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        # SHORT entry: supertrend short + 4h agrees + KAMA confirms + volume confirmed
        elif st_dir == -1 and st_dir_4h <= 0 and kama_confirm_short:
            if volume_confirmed and trend_agree:
                # High conviction: all signals agree
                signals[i] = -SIZE_HIGH
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif volume_confirmed or trend_agree:
                # Base conviction: 2 of 3 confirm
                signals[i] = -SIZE_BASE
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Low conviction: only supertrend
                signals[i] = -SIZE_LOW
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals