#!/usr/bin/env python3
"""
EXPERIMENT #012 - KAMA RSI Pullback with Daily HMA Filter (6h Primary)
==================================================================================================
Hypothesis: 6h primary timeframe offers cleaner signals than 4h (less noise) while maintaining
sufficient trade frequency (unlike 12h). KAMA adapts to volatility better than HMA/EMA, reducing
whipsaws in choppy markets. Combining 6h KAMA trend + 1d HMA filter + RSI pullback entries should
capture medium-term swings more effectively than 4h strategies.

Key innovations:
1. 6h PRIMARY + 1d HTF: Sweet spot between 4h (noisy) and 12h (too few trades)
2. KAMA for adaptive trend: Adjusts smoothing based on market efficiency (ER)
3. 1d HMA as master filter: Stronger trend confirmation than 4h
4. RSI pullback zones: Enter at 35-55 (slightly lower than 40-60 for better entries)
5. Z-score filter: Avoid entries when price is >2 std dev from 20-period mean

Why this should beat #005 (Sharpe=0.537):
- 6h has less noise than 4h, fewer false signals
- KAMA adapts to volatility, better than static HMA in ranging markets
- 1d HMA filter is stronger than 1d Supertrend alone
- Z-score adds mean-reversion filter missing from #005
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_pullback_daily_hma_6h_v1"
timeframe = "6h"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman's Adaptive Moving Average
    Adjusts smoothing based on market efficiency ratio (ER)
    ER = |net change| / sum of absolute changes
    High ER = trending (fast smoothing), Low ER = ranging (slow smoothing)
    """
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    close = np.array(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 0:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
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


def calculate_hma(close, period=21):
    """
    Hull Moving Average - faster than EMA, smoother than SMA
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        result = np.zeros(len(series))
        weights = np.arange(1, window + 1)
        for i in range(window - 1, len(series)):
            result[i] = np.sum(series[i - window + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    close_series = np.array(close)
    wma_half = wma(close_series, half)
    wma_full = wma(close_series, period)
    
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


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
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 6h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_6h = calculate_atr(high, low, close, period=14)
    rsi_6h = calculate_rsi(close, period=14)
    kama_6h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_6h_fast = calculate_kama(close, period=5, fast_period=2, slow_period=15)
    zscore_6h = calculate_zscore(close, period=20)
    supertrend_6h, st_trend_6h = calculate_supertrend(high, low, close, atr_6h, multiplier=3.0)
    
    # ========== 1d INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Daily HMA for trend direction (stronger than Supertrend alone)
        hma_1d = calculate_hma(close_1d, period=21)
        atr_1d = calculate_atr(high_1d, low_1d, close_1d, period=14)
        _, st_trend_1d = calculate_supertrend(high_1d, low_1d, close_1d, atr_1d, multiplier=3.0)
        
        # Align to 6h timeframe (auto shift for completed bars)
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
        st_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, st_trend_1d)
        
    except Exception:
        hma_1d_aligned = np.zeros(n)
        st_trend_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE (max 0.35)
    SIZE_BASE = 0.20
    SIZE_HIGH = 0.35
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    # RSI pullback zones (slightly lower than 40-60 for better entries)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score filter
    ZSCORE_MAX = 2.0
    
    first_valid = max(100, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_6h[i]) or atr_6h[i] == 0 or np.isnan(rsi_6h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_6h[i]
        rsi_val = rsi_6h[i]
        st_trend_val = st_trend_6h[i]
        kama_val = kama_6h[i]
        kama_fast_val = kama_6h_fast[i]
        zscore_val = zscore_6h[i]
        
        # 1d trend filters (MASTER FILTER)
        hma_1d_val = hma_1d_aligned[i]
        st_trend_1d_val = st_trend_1d_aligned[i]
        
        # Determine daily trend direction
        daily_trend = 0
        if hma_1d_val > 0 and price > hma_1d_val:
            daily_trend = 1
        elif hma_1d_val > 0 and price < hma_1d_val:
            daily_trend = -1
        
        if st_trend_1d_val == 1:
            daily_trend = max(daily_trend, 1)
        elif st_trend_1d_val == -1:
            daily_trend = min(daily_trend, -1)
        
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
                    signals[i] = SIZE_BASE
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
                    signals[i] = -SIZE_BASE
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
        # Z-score filter: avoid entries when price is extreme (>2 std dev)
        zscore_filter = abs(zscore_val) < ZSCORE_MAX
        
        # LONG: Daily trend up + 6h Supertrend up + RSI pullback + Z-score OK
        long_condition = (
            daily_trend == 1 and
            st_trend_val == 1 and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            kama_fast_val > kama_val and
            zscore_filter
        )
        
        # SHORT: Daily trend down + 6h Supertrend down + RSI pullback + Z-score OK
        short_condition = (
            daily_trend == -1 and
            st_trend_val == -1 and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            kama_fast_val < kama_val and
            zscore_filter
        )
        
        # Determine position size based on conviction
        high_conviction_long = long_condition and st_trend_1d_val == 1 and hma_1d_val > 0
        high_conviction_short = short_condition and st_trend_1d_val == -1 and hma_1d_val > 0
        
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