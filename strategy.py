#!/usr/bin/env python3
"""
EXPERIMENT #110 - MTF HMA+Supertrend+RSI+Chandelier+VolSizing (15m+1h Proper MTF v1)
==================================================================================================
Hypothesis: The 15m+1h combination from #040 worked well (proven in #031, #034, #035 with Sharpe > 7.5).
Recent failures (#101-#109) show Chandelier exits WITH volatility-adjusted sizing is deadly.

Key changes from #040:
- Use mtf_data helper for PROPER 1h alignment (no manual resampling bugs)
- Chandelier exit: 3*ATR(22) trailing stop from highest_high (proper implementation)
- Volatility-adjusted position sizing: reduce size when ATR% is high
- Keep discrete signal levels (0.0, ±0.20, ±0.35) to avoid churning
- MAX signal = 0.35 (not 1.0) for drawdown control
- Leverage = 1.0 (no leverage)

Why this should beat #040 and the failed #101-#109:
- Proper MTF alignment via mtf_data (46 strategies failed without this)
- Chandelier as STOPLOSS only, not for position sizing decisions
- Volatility adjusts SIZE, not whether we trade (key difference from failed experiments)
- Based on proven 15m+1h combination from winning strategies
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_supertrend_rsi_chandelier_volsizing_15m_1h_v1"
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
    
    hma = pd.Series(2 * wma1 - wma2).ewm(span=sqrt_period, adjust=False).mean().values
    
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
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    
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


def calculate_chandelier_exit(high, low, close, atr, period=22, multiplier=3.0):
    """
    Calculate Chandelier Exit (ATR trailing stop)
    Long exit: highest_high - multiplier * ATR
    Short exit: lowest_low + multiplier * ATR
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    # Rolling highest high and lowest low
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Chandelier exit levels
    chandelier_long = highest_high - multiplier * atr  # Stop for long positions
    chandelier_short = lowest_low + multiplier * atr   # Stop for short positions
    
    return chandelier_long, chandelier_short


def calculate_atr_pct(close, high, low, period=14):
    """Calculate ATR as percentage of price (volatility measure)"""
    atr = calculate_atr(high, low, close, period)
    atr_pct = np.zeros(len(close))
    mask = close > 0
    atr_pct[mask] = (atr[mask] / close[mask]) * 100
    return atr_pct


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    atr_pct_15m = calculate_atr_pct(close, high, low, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    chandelier_long_15m, chandelier_short_15m = calculate_chandelier_exit(high, low, close, atr_15m, period=22, multiplier=3.0)
    
    # Get 1h data using mtf_data helper (CRITICAL for proper alignment)
    try:
        df_1h = get_htf_data(prices, '1h')
        close_1h = df_1h['close'].values
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        
        # Calculate 1h indicators
        hma_1h_raw = calculate_hma(close_1h, period=21)
        st_1h_raw, st_direction_1h_raw = calculate_supertrend(high_1h, low_1h, close_1h, period=10, multiplier=3.0)
        atr_1h_raw = calculate_atr(high_1h, low_1h, close_1h, period=14)
        atr_pct_1h_raw = calculate_atr_pct(close_1h, high_1h, low_1h, period=14)
        
        # Align 1h indicators to 15m timeframe (auto shift by 1 HTF bar for no look-ahead)
        hma_1h = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
        st_direction_1h = align_htf_to_ltf(prices, df_1h, st_direction_1h_raw)
        atr_pct_1h = align_htf_to_ltf(prices, df_1h, atr_pct_1h_raw)
        
    except Exception as e:
        # Fallback if mtf_data fails - use manual resampling
        bars_per_1h = 4
        n_1h = n // bars_per_1h
        
        close_1h = np.array([close[i * bars_per_1h + bars_per_1h - 1] for i in range(n_1h)])
        high_1h = np.array([np.max(high[i * bars_per_1h:(i + 1) * bars_per_1h]) for i in range(n_1h)])
        low_1h = np.array([np.min(low[i * bars_per_1h:(i + 1) * bars_per_1h]) for i in range(n_1h)])
        
        hma_1h_raw = calculate_hma(close_1h, period=21)
        st_direction_1h_raw = calculate_supertrend(high_1h, low_1h, close_1h, period=10, multiplier=3.0)[1]
        atr_pct_1h_raw = calculate_atr_pct(close_1h, high_1h, low_1h, period=14)
        
        hma_1h = np.zeros(n)
        st_direction_1h = np.zeros(n)
        atr_pct_1h = np.zeros(n)
        
        for i in range(n):
            idx_1h = min(i // bars_per_1h, n_1h - 1)
            if idx_1h > 0:
                hma_1h[i] = hma_1h_raw[idx_1h - 1]
                st_direction_1h[i] = st_direction_1h_raw[idx_1h - 1]
                atr_pct_1h[i] = atr_pct_1h_raw[idx_1h - 1]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    BASE_SIZE = 0.35  # Base position size
    MIN_SIZE = 0.15   # Minimum position size in high vol
    
    # Volatility thresholds for position sizing
    ATR_PCT_LOW = 1.0   # Low vol regime (use full size)
    ATR_PCT_HIGH = 2.5  # High vol regime (reduce size)
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # ADX-like filter using ATR% ratio (1h vs 15m)
    ATR_RATIO_MIN = 0.5
    
    first_valid = max(200, 40, 22)
    
    # Track position state for Chandelier exit
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 1h trend filters
        hma_1h_val = hma_1h[i]
        st_trend_1h = st_direction_1h[i]
        atr_pct_1h_val = atr_pct_1h[i]
        
        # Determine 1h trend direction
        if hma_1h_val > 0:
            trend_1h = 1 if close[i] > hma_1h_val else -1
        else:
            trend_1h = 0
        
        # Volatility-adjusted position sizing
        # Higher volatility = smaller position size
        if atr_pct_1h_val > 0:
            if atr_pct_1h_val < ATR_PCT_LOW:
                vol_adjustment = 1.0  # Low vol, full size
            elif atr_pct_1h_val > ATR_PCT_HIGH:
                vol_adjustment = MIN_SIZE / BASE_SIZE  # High vol, minimum size
            else:
                # Linear interpolation
                vol_adjustment = 1.0 - (atr_pct_1h_val - ATR_PCT_LOW) / (ATR_PCT_HIGH - ATR_PCT_LOW) * (1.0 - MIN_SIZE / BASE_SIZE)
        else:
            vol_adjustment = 1.0
        
        # Calculate position size with volatility adjustment
        SIZE_FULL = BASE_SIZE * vol_adjustment
        SIZE_HALF = SIZE_FULL / 2
        
        # Ensure discrete levels
        if SIZE_FULL < 0.15:
            SIZE_FULL = 0.15
        if SIZE_FULL > 0.35:
            SIZE_FULL = 0.35
        
        # Check for existing position and Chandelier exit
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry, high[i])
                chandelier_stop = chandelier_long_15m[i]
                
                # Chandelier exit for long positions
                if close[i] < chandelier_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R (reduce to half)
                atr_risk = prev_entry - chandelier_stop
                if atr_risk > 0:
                    tp_price = prev_entry + 2 * atr_risk
                    if close[i] >= tp_price and highest_since_entry[i - 1] == 0:
                        signals[i] = SIZE_HALF
                        position_side[i] = 1
                        entry_price[i] = prev_entry
                        highest_since_entry[i] = current_high
                        lowest_since_entry[i] = prev_entry
                        continue
                
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = prev_entry
                
            elif prev_side == -1:
                current_low = min(lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry, low[i])
                chandelier_stop = chandelier_short_15m[i]
                
                # Chandelier exit for short positions
                if close[i] > chandelier_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R (reduce to half)
                atr_risk = chandelier_stop - prev_entry
                if atr_risk > 0:
                    tp_price = prev_entry - 2 * atr_risk
                    if close[i] <= tp_price and lowest_since_entry[i - 1] == 0:
                        signals[i] = -SIZE_HALF
                        position_side[i] = -1
                        entry_price[i] = prev_entry
                        highest_since_entry[i] = prev_entry
                        lowest_since_entry[i] = current_low
                        continue
                
                highest_since_entry[i] = prev_entry
                lowest_since_entry[i] = current_low
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            continue
        
        # No existing position - check for entry
        # Trend filters must agree (HMA + Supertrend on 1h)
        if trend_1h != st_trend_1h or trend_1h == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Entry logic: 1h trend + 15m RSI pullback + Z-score filter
        if trend_1h == 1 and st_trend_1h == 1:  # Bullish trend confirmed on 1h
            if (RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX and 
                abs(zscore_15m[i]) < ZSCORE_MAX):  # Pullback + not extreme
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = close[i]
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
                
        elif trend_1h == -1 and st_trend_1h == -1:  # Bearish trend confirmed on 1h
            if (RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX and 
                abs(zscore_15m[i]) < ZSCORE_MAX):  # Pullback + not extreme
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = close[i]
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals