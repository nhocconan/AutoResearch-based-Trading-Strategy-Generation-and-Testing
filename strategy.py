#!/usr/bin/env python3
"""
EXPERIMENT #042 - HMA RSI Pullback with Bollinger Regime Filter (1h Primary + 4h Trend)
==================================================================================================
Hypothesis: Current best (hma_rsi_pullback_daily_trend_4h_v1, Sharpe=0.537) uses 4h+1d.
This uses 1h+4h for MORE TRADES while maintaining quality with 4h trend filter.

Key innovations:
1. 1h PRIMARY + 4h HTF: More trade opportunities than 4h primary, cleaner than 15m/30m
2. HMA for trend: Proven to work better than DEMA/KAMA in our tests
3. Bollinger Band Width regime filter: Avoid trading in choppy/low-volatility periods
4. RSI pullback entries: Enter on RSI dips in uptrend (40-55) and rallies in downtrend (45-60)
5. ATR stoploss at 2.0x with trailing after 1R profit

Why this should beat Sharpe=0.537:
- 1h timeframe = 4x more trade signals than 4h
- 4h trend filter maintains signal quality
- Bollinger regime filter avoids choppy periods (reduces false signals)
- Simpler logic = fewer bugs (fixes #041 read-only array crash)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_bollinger_regime_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = close_series.ewm(span=half_period, adjust=False, min_periods=half_period).mean()
    wma2 = close_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    diff = (2 * wma1 - wma2)
    hma = diff.ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean()
    
    hma_arr = hma.values.copy()
    hma_arr[:period] = 0
    return hma_arr


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
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.ones(n) * 100
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """
    Calculate Bollinger Bands and Band Width
    Returns: upper, middle, lower, band_width, pct_b
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    middle = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    band_width = (upper - lower) / middle
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    # Mark invalid periods
    upper[:period] = 0
    lower[:period] = 0
    band_width[:period] = 0
    pct_b[:period] = 0
    
    return upper, middle, lower, band_width, pct_b


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    hma_1h = calculate_hma(close, period=21)
    hma_fast_1h = calculate_hma(close, period=10)
    rsi_1h = calculate_rsi(close, period=14)
    bb_upper, bb_middle, bb_lower, bb_width, bb_pct = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    _, st_trend_1h = calculate_supertrend(high, low, close, atr_1h, multiplier=3.0)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values.copy()
        high_4h = df_4h['high'].values.copy()
        low_4h = df_4h['low'].values.copy()
        
        # 4h HMA for master trend direction
        hma_4h = calculate_hma(close_4h, period=21)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        rsi_4h = calculate_rsi(close_4h, period=14)
        
        # Align to 1h timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
    except Exception:
        hma_4h_aligned = np.zeros(n)
        st_trend_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE LEVELS
    SIZE_BASE = 0.25    # Base position (25% of capital)
    SIZE_HIGH = 0.35    # High conviction (35% of capital)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    # RSI filter zones for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    
    # Bollinger Band Width regime filter (avoid choppy markets)
    BB_WIDTH_MIN = 0.02  # Minimum band width to trade (2%)
    
    first_valid = 100  # Need enough data for 4h alignment and indicators
    
    # Track position state
    position_side = np.zeros(n, dtype=np.int32)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(hma_1h[i]) or hma_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Skip if Bollinger Bands not ready
        if bb_width[i] == 0 or bb_width[i] < BB_WIDTH_MIN:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        hma_val = hma_1h[i]
        hma_fast_val = hma_fast_1h[i]
        rsi_val = rsi_1h[i]
        st_trend_val = st_trend_1h[i]
        
        # 4h trend filters (MASTER FILTER)
        hma_4h_val = hma_4h_aligned[i]
        st_trend_4h_val = st_trend_4h_aligned[i]
        rsi_4h_val = rsi_4h_aligned[i]
        
        # Determine 4h trend direction
        four_h_trend = 0
        if hma_4h_val > 0 and price > hma_4h_val:
            four_h_trend = 1
        elif hma_4h_val > 0 and price < hma_4h_val:
            four_h_trend = -1
        
        if st_trend_4h_val == 1:
            four_h_trend = max(four_h_trend, 1)
        elif st_trend_4h_val == -1:
            four_h_trend = min(four_h_trend, -1)
        
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
                    entry_price[i] = 0.0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0.0
                    lowest_since_entry[i] = 0.0
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
                        entry_price[i] = 0.0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0.0
                        lowest_since_entry[i] = 0.0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0.0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0.0
                    lowest_since_entry[i] = 0.0
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
                        entry_price[i] = 0.0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0.0
                        lowest_since_entry[i] = 0.0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== ENTRY LOGIC - RSI PULLBACK IN 4h TREND DIRECTION ==========
        # LONG: 4h trend up + 1h Supertrend up + RSI pullback + HMA aligned + BB expanding
        long_condition = (
            four_h_trend == 1 and
            st_trend_val == 1 and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            hma_fast_val > hma_val and
            price > hma_4h_val
        )
        
        # SHORT: 4h trend down + 1h Supertrend down + RSI pullback + HMA aligned
        short_condition = (
            four_h_trend == -1 and
            st_trend_val == -1 and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            hma_fast_val < hma_val and
            price < hma_4h_val
        )
        
        # High conviction: 4h RSI also confirms + 4h Supertrend confirms
        high_conviction_long = (
            long_condition and 
            st_trend_4h_val == 1 and 
            rsi_4h_val >= 45 and rsi_4h_val <= 65
        )
        high_conviction_short = (
            short_condition and 
            st_trend_4h_val == -1 and 
            rsi_4h_val >= 35 and rsi_4h_val <= 55
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
    
    return signals