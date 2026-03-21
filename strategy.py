#!/usr/bin/env python3
"""
EXPERIMENT #013 - KAMA Adaptive Trend with Triple Timeframe Filter (1h Primary)
==================================================================================================
Hypothesis: Current best uses 4h+1d. This uses 1h+4h+1d triple timeframe for more entry opportunities
while maintaining strict trend alignment. KAMA adapts to market regime (fast in trends, slow in chop).

Key innovations:
1. Triple MTF: 1h entries + 4h trend + 1d master filter (stronger than dual MTF)
2. KAMA adaptive trend: Automatically adjusts sensitivity based on volatility/efficiency ratio
3. BBW regime filter: Only trade when Bollinger Bands are expanding (avoid squeeze/chop)
4. Z-score entry timing: Enter when price is 1-2 std dev from mean in trend direction
5. Volume confirmation: Require above-average volume on entry bars

Why this should beat hma_rsi_pullback_daily_trend_4h_v1 (Sharpe=0.537):
- More entry signals (1h vs 4h primary) while keeping strict HTF filters
- KAMA adapts better than HMA in changing regimes
- BBW filter avoids low-volatility chop that causes whipsaws
- Z-score provides better entry timing than RSI alone
- Triple timeframe alignment reduces false signals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_triple_mtf_bbw_zscore_1h_v1"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise: fast in trends, slow in chop
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = sum(abs(close[j] - close[j - 1]) for j in range(i - period + 1, i + 1))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bbw = (upper - lower) / sma
    
    # Handle NaN/inf
    bbw = np.nan_to_num(bbw, nan=0.0, posinf=0.0, neginf=0.0)
    
    return upper, lower, bbw


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = std > 0
    zscore[mask] = (close[mask] - sma[mask]) / std[mask]
    
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


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile rank over lookback period"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback, n):
        window = bbw[i - lookback:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / len(window)
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    kama_1h = calculate_kama(close, period=10, fast=2, slow=30)
    kama_1h_fast = calculate_kama(close, period=5, fast=2, slow=15)
    supertrend_1h, st_trend_1h = calculate_supertrend(high, low, close, atr_1h, multiplier=3.0)
    bb_upper_1h, bb_lower_1h, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    zscore_1h = calculate_zscore(close, period=20)
    bbw_pct_1h = calculate_bbw_percentile(bbw_1h, lookback=100)
    
    # Volume MA for confirmation
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ========== 4h INDICATORS (INTERMEDIATE TREND) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        kama_4h = calculate_kama(close_4h, period=10, fast=2, slow=30)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        
        # Align to 1h timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
        st_trend_4h_aligned = np.zeros(n)
    
    # ========== 1d INDICATORS (MASTER TREND FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        kama_1d = calculate_kama(close_1d, period=10, fast=2, slow=30)
        atr_1d = calculate_atr(high_1d, low_1d, close_1d, period=14)
        _, st_trend_1d = calculate_supertrend(high_1d, low_1d, close_1d, atr_1d, multiplier=3.0)
        
        # Align to 1h timeframe (auto shift for completed bars)
        kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
        st_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, st_trend_1d)
        
    except Exception:
        kama_1d_aligned = np.zeros(n)
        st_trend_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE & DISCRETE
    SIZE_BASE = 0.20   # Base position (20%)
    SIZE_HIGH = 0.30   # High conviction (30%)
    SIZE_MAX = 0.35    # Maximum position
    
    # ATR stoploss
    ATR_STOP_MULT = 2.5
    
    # RSI pullback zones
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score entry zones
    ZSCORE_LONG_MIN = -1.5
    ZSCORE_LONG_MAX = 0.5
    ZSCORE_SHORT_MIN = -0.5
    ZSCORE_SHORT_MAX = 1.5
    
    # BBW percentile filter (avoid squeeze < 0.2, avoid extreme > 0.9)
    BBW_PCT_MIN = 0.20
    BBW_PCT_MAX = 0.85
    
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
        zscore_val = zscore_1h[i]
        bbw_pct_val = bbw_pct_1h[i]
        vol_ratio = volume[i] / vol_ma_1h[i] if vol_ma_1h[i] > 0 else 1.0
        
        # 4h trend filters
        kama_4h_val = kama_4h_aligned[i]
        st_trend_4h_val = st_trend_4h_aligned[i]
        
        # 1d trend filters (MASTER FILTER)
        kama_1d_val = kama_1d_aligned[i]
        st_trend_1d_val = st_trend_1d_aligned[i]
        
        # Determine trend directions
        # 1d master trend
        daily_trend = 0
        if kama_1d_val > 0 and price > kama_1d_val:
            daily_trend = 1
        elif kama_1d_val > 0 and price < kama_1d_val:
            daily_trend = -1
        
        if st_trend_1d_val == 1:
            daily_trend = max(daily_trend, 1)
        elif st_trend_1d_val == -1:
            daily_trend = min(daily_trend, -1)
        
        # 4h intermediate trend
        h4_trend = 0
        if kama_4h_val > 0 and price > kama_4h_val:
            h4_trend = 1
        elif kama_4h_val > 0 and price < kama_4h_val:
            h4_trend = -1
        
        if st_trend_4h_val == 1:
            h4_trend = max(h4_trend, 1)
        elif st_trend_4h_val == -1:
            h4_trend = min(h4_trend, -1)
        
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
        
        # ========== REGIME FILTER (BBW PERCENTILE) ==========
        # Only trade when BBW is in normal range (not squeeze, not extreme expansion)
        regime_ok = bbw_pct_val >= BBW_PCT_MIN and bbw_pct_val <= BBW_PCT_MAX
        
        # Volume confirmation (above 0.8x average)
        volume_ok = vol_ratio >= 0.8
        
        if not regime_ok or not volume_ok:
            signals[i] = 0.0
            continue
        
        # ========== ENTRY LOGIC - TRIPLE TIMEFRAME ALIGNMENT ==========
        # LONG: 1d trend up + 4h trend up + 1h Supertrend up + RSI pullback + Z-score entry
        long_condition = (
            daily_trend == 1 and
            h4_trend == 1 and
            st_trend_val == 1 and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            zscore_val >= ZSCORE_LONG_MIN and zscore_val <= ZSCORE_LONG_MAX and
            kama_fast_val > kama_val  # Fast KAMA above slow KAMA
        )
        
        # SHORT: 1d trend down + 4h trend down + 1h Supertrend down + RSI pullback + Z-score entry
        short_condition = (
            daily_trend == -1 and
            h4_trend == -1 and
            st_trend_val == -1 and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            zscore_val >= ZSCORE_SHORT_MIN and zscore_val <= ZSCORE_SHORT_MAX and
            kama_fast_val < kama_val  # Fast KAMA below slow KAMA
        )
        
        # Determine position size based on conviction
        # High conviction: all three timeframes align + strong volume
        high_conviction_long = long_condition and st_trend_1d_val == 1 and st_trend_4h_val == 1 and vol_ratio >= 1.2
        high_conviction_short = short_condition and st_trend_1d_val == -1 and st_trend_4h_val == -1 and vol_ratio >= 1.2
        
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