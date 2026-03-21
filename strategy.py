#!/usr/bin/env python3
"""
EXPERIMENT #072 - Regime Adaptive MTF KAMA Supertrend RSI (15m + 4h Clean)
==================================================================================================
Hypothesis: Recent ensemble voting strategies (#063-#071) failed due to complexity and improper MTF alignment.
The kept strategies (#060-#062, #066) used simpler 15m+4h combinations with Sharpe 0.14-0.22.
Current best (Sharpe=3.653) uses Supertrend+MACD+BBW+RSI on 15m/1h/4h.

Key changes from #040:
- USE mtf_data helper for PROPER 4h alignment (CRITICAL - 46 strategies failed without this)
- Regime detection: BBW percentile → trend follow in low vol, mean revert in high vol
- Adaptive sizing: 0.20 base, scale to 0.35 on high confidence (multiple signals agree)
- Simplified logic: KAMA trend + Supertrend stops + RSI entries (no complex voting)
- Timeframe: 15m entries + 4h trend filter (proven in #060-#062)
- Position size: 0.20-0.35 discrete levels (conservative for drawdown control)

Why this should beat #040:
- Proper MTF alignment prevents look-ahead bias (mtf_data helper)
- Regime-adaptive logic reduces losses in choppy markets
- Adaptive sizing increases wins when confidence is high
- Simpler logic = fewer bugs and crashes (learned from #064, #067, #070)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_adaptive_kama_supertrend_rsi_mtf_15m_4h_v1"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # 4h trend filter using mtf_data helper (CRITICAL - proper alignment)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h indicators
    kama_4h = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
    supertrend_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    
    # Align 4h indicators to 15m timeframe (auto shift for completed bars)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    bbw_pct_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_pct_4h)
    
    # Generate signals with regime-adaptive logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_BASE = 0.20
    SIZE_HIGH = 0.35
    SIZE_HALF = 0.10
    
    # RSI thresholds for entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Regime thresholds (BBW percentile)
    LOW_VOL_THRESHOLD = 0.30  # Below 30th percentile = low vol (trend follow)
    HIGH_VOL_THRESHOLD = 0.70  # Above 70th percentile = high vol (mean revert)
    
    first_valid = max(200, 100, 14 * 2, 20, 28)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filters
        kama_4h_val = kama_4h_aligned[i]
        st_4h_val = st_direction_4h_aligned[i]
        bbw_pct_4h_val = bbw_pct_4h_aligned[i]
        
        # Skip if 4h data not available
        if kama_4h_val == 0 or st_4h_val == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Determine 4h trend direction
        trend_4h = 0
        if close[i] > kama_4h_val and st_4h_val == 1:
            trend_4h = 1
        elif close[i] < kama_4h_val and st_4h_val == -1:
            trend_4h = -1
        
        # 15m regime detection
        bbw_pct_15m_val = bbw_pct_15m[i]
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, close[i])
                current_low = min(prev_low, close[i]) if prev_low > 0 else close[i]
            else:
                current_high = max(prev_high, close[i]) if prev_high > 0 else close[i]
                current_low = min(prev_low, close[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
                    if close[i] > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
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
        
        # Regime-adaptive entry logic
        rsi_val = rsi_15m[i]
        
        # LOW VOLATILITY REGIME (trend following)
        if bbw_pct_15m_val < LOW_VOL_THRESHOLD and bbw_pct_4h_val < LOW_VOL_THRESHOLD:
            # Trend follow: enter in direction of 4h trend on pullbacks
            if trend_4h == 1:
                if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                    # High confidence: 4h trend + 15m supertrend agree
                    if st_direction_15m[i] == 1:
                        signals[i] = SIZE_HIGH
                    else:
                        signals[i] = SIZE_BASE
                    position_side[i] = 1 if signals[i] > 0 else 0
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
                    
            elif trend_4h == -1:
                if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                    # High confidence: 4h trend + 15m supertrend agree
                    if st_direction_15m[i] == -1:
                        signals[i] = -SIZE_HIGH
                    else:
                        signals[i] = -SIZE_BASE
                    position_side[i] = -1 if signals[i] < 0 else 0
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
        
        # HIGH VOLATILITY REGIME (mean reversion)
        elif bbw_pct_15m_val > HIGH_VOL_THRESHOLD or bbw_pct_4h_val > HIGH_VOL_THRESHOLD:
            # Mean revert: fade extremes against 4h trend
            if trend_4h == 1:
                # In uptrend, buy deep pullbacks (RSI < 35)
                if rsi_val < 30:
                    signals[i] = SIZE_BASE
                    position_side[i] = 1
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
                    
            elif trend_4h == -1:
                # In downtrend, sell sharp rallies (RSI > 70)
                if rsi_val > 70:
                    signals[i] = -SIZE_BASE
                    position_side[i] = -1
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
        
        # NORMAL VOLATILITY REGIME (standard trend follow)
        else:
            if trend_4h == 1:
                if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and st_direction_15m[i] == 1:
                    signals[i] = SIZE_BASE
                    position_side[i] = 1
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
                    
            elif trend_4h == -1:
                if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and st_direction_15m[i] == -1:
                    signals[i] = -SIZE_BASE
                    position_side[i] = -1
                    entry_price[i] = close[i]
                    tp_triggered[i] = 0
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
        
        if position_side[i] == 0:
            signals[i] = 0.0
    
    return signals