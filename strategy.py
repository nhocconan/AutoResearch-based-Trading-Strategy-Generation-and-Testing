#!/usr/bin/env python3
"""
EXPERIMENT #054 - Donchian Breakout + ADX Regime + RSI Momentum (30m Primary)
==================================================================================================
Hypothesis: Current best (Sharpe=0.563) uses BB width for regime detection. 
This strategy uses ADX for trend strength + Donchian channels for breakouts.

Key innovations:
1. ADX regime detection: ADX>25 = strong trend (follow breakouts), ADX<20 = choppy (avoid)
2. Donchian channel breakouts: 20-period high/low for clean entry signals
3. RSI momentum confirmation: RSI>55 for longs, RSI<45 for shorts (not just extremes)
4. Multi-timeframe: 30m entries + 4h trend + 1d regime filter
5. Adaptive sizing: 0.25 base, 0.35 when ADX confirms strong trend

Why this should beat current best (Sharpe=0.563):
- Donchian breakouts capture clean trend moves without whipsaw
- ADX filters out choppy markets where trend strategies fail
- 30m timeframe captures more opportunities than 1h
- 1d regime filter avoids trading against major trend
- RSI momentum confirmation reduces false breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_adx_rsi_mtf_30m_4h_1d_v1"
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
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    # Smooth over period
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    mask = tr_smooth > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle


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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    
    # WMA with period/2
    wma_half = close_series.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean().values
    
    # WMA with period
    wma_full = close_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # 2*WMA_half - WMA_full
    raw_hma = 2 * wma_half - wma_full
    
    # Smooth with sqrt(period) WMA
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean().values
    
    return hma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 30m INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_30m = calculate_atr(high, low, close, period=14)
    adx_30m = calculate_adx(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, period=20)
    hma_30m = calculate_hma(close, period=21)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators
        adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
        rsi_4h = calculate_rsi(close_4h, period=14)
        hma_4h = calculate_hma(close_4h, period=21)
        
        # Align to 30m timeframe (auto shift for completed bars)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        
    except Exception:
        adx_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
        hma_4h_aligned = np.zeros(n)
    
    # ========== 1d INDICATORS (REGIME FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # 1d indicators
        adx_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
        hma_1d = calculate_hma(close_1d, period=21)
        
        # Align to 30m timeframe
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
        
    except Exception:
        adx_1d_aligned = np.zeros(n)
        hma_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.25   # Base position
    SIZE_HIGH = 0.35   # High conviction (all signals align)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.5
    
    # RSI momentum thresholds
    RSI_LONG_THRESHOLD = 55
    RSI_SHORT_THRESHOLD = 45
    
    # ADX thresholds
    ADX_STRONG = 25    # Strong trend
    ADX_WEAK = 20      # Weak/choppy
    
    first_valid = max(200, 150)
    
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
        adx_val = adx_30m[i]
        dc_upper = donchian_upper[i]
        dc_lower = donchian_lower[i]
        hma_val = hma_30m[i]
        
        # 4h trend filters
        adx_4h_val = adx_4h_aligned[i]
        rsi_4h_val = rsi_4h_aligned[i]
        hma_4h_val = hma_4h_aligned[i]
        
        # 1d regime filters
        adx_1d_val = adx_1d_aligned[i]
        hma_1d_val = hma_1d_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if hma_4h_val > 0 and price > hma_4h_val:
            trend_4h = 1
        elif hma_4h_val > 0 and price < hma_4h_val:
            trend_4h = -1
        
        # Determine 1d regime
        trend_1d = 0
        if hma_1d_val > 0 and price > hma_1d_val:
            trend_1d = 1
        elif hma_1d_val > 0 and price < hma_1d_val:
            trend_1d = -1
        
        # Determine if trend is strong (ADX filter)
        is_strong_trend_30m = adx_val > ADX_STRONG
        is_strong_trend_4h = adx_4h_val > ADX_STRONG
        is_choppy_30m = adx_val < ADX_WEAK
        
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
                    signals[i] = SIZE_BASE  # Reduce to half
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
                    signals[i] = -SIZE_BASE  # Reduce to half
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
        
        # ========== ENTRY LOGIC - DONCHIAN BREAKOUT + ADX + RSI ==========
        
        # Donchian breakout signals
        breakout_long = price > dc_upper and dc_upper > 0
        breakout_short = price < dc_lower and dc_lower > 0
        
        # RSI momentum confirmation
        rsi_confirms_long = rsi_val > RSI_LONG_THRESHOLD
        rsi_confirms_short = rsi_val < RSI_SHORT_THRESHOLD
        
        # 4h trend filter
        trend_4h_confirms_long = trend_4h != -1  # Not bearish on 4h
        trend_4h_confirms_short = trend_4h != 1  # Not bullish on 4h
        
        # 1d regime filter (only trade with daily trend)
        regime_confirms_long = trend_1d != -1  # Not bearish on 1d
        regime_confirms_short = trend_1d != 1  # Not bullish on 1d
        
        # ADX trend strength filter
        adx_confirms_long = is_strong_trend_30m or is_strong_trend_4h
        adx_confirms_short = is_strong_trend_30m or is_strong_trend_4h
        
        # Avoid trading in choppy markets (ADX < 20 on both timeframes)
        avoid_choppy = not (is_choppy_30m and adx_4h_val < ADX_WEAK)
        
        # LONG conditions
        long_condition = (
            breakout_long and
            rsi_confirms_long and
            trend_4h_confirms_long and
            regime_confirms_long and
            avoid_choppy
        )
        
        # SHORT conditions
        short_condition = (
            breakout_short and
            rsi_confirms_short and
            trend_4h_confirms_short and
            regime_confirms_short and
            avoid_choppy
        )
        
        # High conviction: ADX confirms strong trend + all timeframes align
        high_conviction_long = (
            long_condition and
            is_strong_trend_30m and
            trend_4h == 1 and
            trend_1d == 1
        )
        
        high_conviction_short = (
            short_condition and
            is_strong_trend_30m and
            trend_4h == -1 and
            trend_1d == -1
        )
        
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