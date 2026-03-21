#!/usr/bin/env python3
"""
EXPERIMENT #037 - KAMA Supertrend RSI Pullback with Volatility Filter (1h Primary)
==================================================================================================
Hypothesis: Best kept strategies use 1h_4h MTF (#027, #030, #034 all Sharpe > 0.4).
KAMA adapts to volatility better than HMA in ranging markets. Adding ATR percentile
volatility filter avoids entries during extreme volatility (whipsaw) or low volatility
(no momentum). This should beat current best (hma_rsi_pullback_daily_trend_4h_v1 Sharpe=0.537).

Key innovations:
1. 1h PRIMARY + 4h HTF: Proven combination in multiple kept strategies
2. KAMA for adaptive trend: Adjusts smoothing based on market efficiency
3. Supertrend for direction: Clean trend signals with ATR-based stops
4. RSI pullback: Enter on 45-55 zone (neutral, not extreme)
5. Volatility filter: Only trade when ATR is in 30th-70th percentile (avoid extremes)
6. Conservative sizing: 0.25 base, 0.30 high conviction, 2.0 ATR stoploss
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_supertrend_rsi_volfilter_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based stops
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    if n < len(atr) or len(atr) == 0:
        return np.zeros(n), np.zeros(n)
    
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(n):
        if atr[i] == 0:
            continue
        upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
    
    first_valid = np.where(atr > 0)[0]
    if len(first_valid) == 0:
        return supertrend, trend
    
    start_idx = first_valid[0]
    supertrend[start_idx] = upper_band[start_idx]
    trend[start_idx] = 1
    
    for i in range(start_idx + 1, n):
        if atr[i] == 0:
            supertrend[i] = supertrend[i - 1]
            trend[i] = trend[i - 1]
            continue
        
        if trend[i - 1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = max(supertrend[i - 1], lower_band[i])
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = min(supertrend[i - 1], upper_band[i])
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, trend


def calculate_atr_percentile(atr, lookback=100):
    """Calculate ATR percentile rank over lookback period"""
    n = len(atr)
    percentile = np.zeros(n)
    
    for i in range(lookback, n):
        window = atr[i - lookback:i + 1]
        rank = np.sum(window < atr[i]) / lookback
        percentile[i] = rank
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_1h_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    supertrend_1h, st_trend_1h = calculate_supertrend(high, low, close, atr_1h, multiplier=3.0)
    atr_pct_1h = calculate_atr_percentile(atr_1h, lookback=100)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h KAMA for trend direction
        kama_4h = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        
        # Align to 1h timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
        st_trend_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.25   # Base position
    SIZE_HIGH = 0.30   # High conviction
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    # RSI pullback zones (neutral zone, not extreme)
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 55
    
    # Volatility filter - only trade in normal volatility
    VOL_MIN_PCT = 0.30
    VOL_MAX_PCT = 0.70
    
    first_valid = max(150, 100)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        rsi_val = rsi_1h[i]
        st_trend_val = st_trend_1h[i]
        kama_val = kama_1h[i]
        kama_fast_val = kama_1h_fast[i]
        vol_pct = atr_pct_1h[i]
        
        # 4h trend filters (MASTER FILTER)
        kama_4h_val = kama_4h_aligned[i]
        st_trend_4h_val = st_trend_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if kama_4h_val > 0 and price > kama_4h_val:
            trend_4h = 1
        elif kama_4h_val > 0 and price < kama_4h_val:
            trend_4h = -1
        
        if st_trend_4h_val == 1:
            trend_4h = max(trend_4h, 1)
        elif st_trend_4h_val == -1:
            trend_4h = min(trend_4h, -1)
        
        # Volatility filter - skip extreme volatility
        vol_ok = (vol_pct >= VOL_MIN_PCT and vol_pct <= VOL_MAX_PCT)
        
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
            
            # Stoploss check (2.0*ATR)
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
                    signals[i] = SIZE_BASE / 2  # Reduce to half
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
                    signals[i] = -SIZE_BASE / 2  # Reduce to half
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
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== ENTRY LOGIC - RSI PULLBACK IN TREND DIRECTION ==========
        # LONG: 4h trend up + 1h Supertrend up + RSI pullback (45-55) + KAMA fast > slow + vol ok
        long_condition = (
            trend_4h == 1 and
            st_trend_val == 1 and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            kama_fast_val > kama_val and
            vol_ok
        )
        
        # SHORT: 4h trend down + 1h Supertrend down + RSI pullback (45-55) + KAMA fast < slow + vol ok
        short_condition = (
            trend_4h == -1 and
            st_trend_val == -1 and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            kama_fast_val < kama_val and
            vol_ok
        )
        
        # Determine position size based on conviction
        # High conviction: 4h Supertrend aligns with entry direction
        high_conviction_long = long_condition and st_trend_4h_val == 1
        high_conviction_short = short_condition and st_trend_4h_val == -1
        
        if long_condition:
            size = SIZE_HIGH if high_conviction_long else SIZE_BASE
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            size = SIZE_HIGH if high_conviction_short else SIZE_BASE
            signals[i] = -size
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