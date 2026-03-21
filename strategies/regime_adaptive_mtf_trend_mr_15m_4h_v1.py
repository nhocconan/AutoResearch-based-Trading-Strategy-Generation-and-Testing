#!/usr/bin/env python3
"""
EXPERIMENT #066 - Regime-Adaptive MTF Strategy (15m + 4h)
==================================================================================================
Hypothesis: Complex ensemble voting failed (#054, #058, #063, #065). Simpler regime detection worked (#055, #060, #061, #062).

Key insight from failures:
- Ensemble voting with 3+ strategies caused negative Sharpe (conflicting signals)
- Too many timeframes (15m+1h+4h) caused timeout (#064) or poor alignment
- Regime detection alone (BBW percentile) showed promise (#055 Sharpe=0.164)

New approach:
- Regime detection: BBW percentile on 4h → LOW vol = trend follow, HIGH vol = mean revert
- Timeframes: 15m (entry) + 4h (regime & trend) - skip 1h to reduce complexity
- Use mtf_data helper properly (get_htf_data, align_htf_to_ltf)
- Adaptive position sizing: higher confidence in regime = larger position
- Discrete signal levels: 0.0, ±0.20, ±0.35 (minimize churn costs)

Why this should beat #040 (Sharpe unknown) and #065 (Sharpe=-7.9):
- Proper MTF alignment using mtf_data (46 strategies failed without this)
- Regime-adaptive logic (trend vs mean reversion based on volatility)
- Simpler than failed ensemble strategies
- Based on #055 success pattern (regime_adaptive_mtf_bbw_rsi_hma_15m_4h_v1)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_adaptive_mtf_trend_mr_15m_4h_v1"
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = rolling_std > 0
    zscore[mask] = (close[mask] - rolling_mean[mask]) / rolling_std[mask]
    
    return zscore


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    mid = (high + low) / 2
    upper_band = mid + multiplier * atr
    lower_band = mid - multiplier * atr
    
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
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    bbw = np.zeros(n)
    mask = middle > 0
    bbw[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
    return upper, middle, lower, bbw


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile rank for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i]) / len(window)
        percentile[i] = rank
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === 15m indicators (entry timing) ===
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # === 4h indicators (regime & trend) using mtf_data helper ===
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h indicators
        hma_4h = calculate_hma(c_4h, period=21)
        st_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
        _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
        bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_dir_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        bbw_pct_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_pct_4h)
        
    except Exception as e:
        # Fallback: use 15m data only if 4h not available
        hma_4h_aligned = hma_15m
        st_dir_4h_aligned = st_direction_15m
        bbw_pct_4h_aligned = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # === Regime detection ===
    # LOW volatility (BBW percentile < 0.3): Trend following mode
    # HIGH volatility (BBW percentile > 0.7): Mean reversion mode
    # MEDIUM volatility (0.3-0.7): No trade or reduced size
    REGIME_LOW_THRESHOLD = 0.30
    REGIME_HIGH_THRESHOLD = 0.70
    
    # === Position sizing (discrete levels) ===
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.10
    
    # === Entry thresholds ===
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    ZSCORE_MR_THRESHOLD = 1.5
    ATR_STOP_MULT = 2.5
    
    # === Initialize tracking arrays ===
    signals = np.zeros(n)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    first_valid = max(200, 100)  # Need enough data for BBW percentile
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(bbw_pct_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        regime = bbw_pct_4h_aligned[i]
        st_trend_4h = st_dir_4h_aligned[i]
        hma_4h_val = hma_4h_aligned[i]
        
        # === Check existing positions (stoploss & take profit) ===
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
            
            # Stoploss check
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        
        # LOW volatility regime: Trend following
        if regime < REGIME_LOW_THRESHOLD:
            # 4h Supertrend determines trend direction
            if st_trend_4h == 1 and price > hma_4h_val:
                # 15m Supertrend confirms entry
                if st_direction_15m[i] == 1:
                    # RSI pullback filter
                    if RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX:
                        signals[i] = SIZE_FULL
                        position_side[i] = 1
                        entry_price[i] = price
                        tp_triggered[i] = 0
                        highest_since_entry[i] = price
                        lowest_since_entry[i] = price
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
                        
            elif st_trend_4h == -1 and price < hma_4h_val:
                # 15m Supertrend confirms entry
                if st_direction_15m[i] == -1:
                    # RSI pullback filter
                    if RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX:
                        signals[i] = -SIZE_FULL
                        position_side[i] = -1
                        entry_price[i] = price
                        tp_triggered[i] = 0
                        highest_since_entry[i] = price
                        lowest_since_entry[i] = price
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        # HIGH volatility regime: Mean reversion
        elif regime > REGIME_HIGH_THRESHOLD:
            zscore_val = zscore_15m[i]
            
            # Mean reversion: buy when oversold, sell when overbought
            if zscore_val < -ZSCORE_MR_THRESHOLD and rsi_15m[i] < 40:
                signals[i] = SIZE_HALF  # Reduced size in high vol
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
            elif zscore_val > ZSCORE_MR_THRESHOLD and rsi_15m[i] > 60:
                signals[i] = -SIZE_HALF  # Reduced size in high vol
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        # MEDIUM volatility: No trade or very small positions
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals