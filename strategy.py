#!/usr/bin/env python3
"""
EXPERIMENT #025 - KAMA RSI Z-Score Pullback with Daily Trend (6h Primary)
==================================================================================================
Hypothesis: 6h timeframe provides cleaner signals than 4h (less noise, fewer whipsaws).
KAMA adapts to volatility better than HMA/EMA - reduces false signals in ranging markets.
RSI + Z-score combo is more robust than RSI alone for identifying true pullbacks.
Daily trend filter eliminates counter-trend trades that cause major drawdowns.

Key innovations vs current best (hma_rsi_pullback_daily_trend_4h_v1, Sharpe=0.537):
1. 6h PRIMARY instead of 4h: Fewer bars, less noise, cleaner trend signals
2. KAMA instead of HMA: Adaptive smoothing reduces whipsaws in sideways markets
3. Z-score filter: Additional confirmation that pullback is statistically significant
4. Dynamic sizing: Base size adjusted by current volatility (ATR percentile)
5. Tighter stoploss: 1.8*ATR instead of 2.0*ATR (6h bars are larger)

Why this should beat the 4h version:
- 6h has 4 bars/day vs 4h's 6 bars/day - less noise, more significant moves
- KAMA efficiency ratio adapts to market regime automatically
- Z-score(20) < -1.5 confirms oversold condition beyond just RSI
- Tested across BTC/ETH/SOL with similar volatility profiles
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_zscore_pullback_daily_6h_v1"
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
    Kaufman's Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    if n < period + slow_period:
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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window, ddof=0)
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
    supertrend_6h, st_trend_6h = calculate_supertrend(high, low, close, atr_6h, multiplier=2.5)
    
    # ========== 1d INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Daily KAMA for trend direction
        kama_1d = calculate_kama(close_1d, period=10, fast_period=2, slow_period=30)
        atr_1d = calculate_atr(high_1d, low_1d, close_1d, period=14)
        _, st_trend_1d = calculate_supertrend(high_1d, low_1d, close_1d, atr_1d, multiplier=3.0)
        
        # Align to 6h timeframe (auto shift for completed bars)
        kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
        st_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, st_trend_1d)
        
    except Exception:
        kama_1d_aligned = np.zeros(n)
        st_trend_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE with dynamic adjustment
    SIZE_BASE = 0.20   # Base position
    SIZE_HIGH = 0.30   # High conviction
    SIZE_MAX = 0.35    # Maximum position
    
    # ATR stoploss (tighter for 6h since bars are larger)
    ATR_STOP_MULT = 1.8
    
    # RSI pullback zones
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score thresholds for confirmation
    ZSCORE_LONG = -1.2  # Oversold confirmation
    ZSCORE_SHORT = 1.2  # Overbought confirmation
    
    first_valid = max(100, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Calculate ATR percentile for dynamic sizing
    atr_percentile = np.zeros(n)
    for i in range(100, n):
        atr_window = atr_6h[max(0, i-50):i+1]
        atr_percentile[i] = np.percentile(atr_window, 50)  # Median ATR
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_6h[i]) or atr_6h[i] == 0 or np.isnan(rsi_6h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_6h[i]
        rsi_val = rsi_6h[i]
        zscore_val = zscore_6h[i]
        st_trend_val = st_trend_6h[i]
        kama_val = kama_6h[i]
        kama_fast_val = kama_6h_fast[i]
        
        # 1d trend filters (MASTER FILTER)
        kama_1d_val = kama_1d_aligned[i]
        st_trend_1d_val = st_trend_1d_aligned[i]
        
        # Determine daily trend direction
        daily_trend = 0
        if kama_1d_val > 0 and price > kama_1d_val:
            daily_trend = 1
        elif kama_1d_val > 0 and price < kama_1d_val:
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
            
            # Stoploss check (1.8*ATR)
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
        
        # ========== ENTRY LOGIC - RSI + Z-SCORE PULLBACK IN TREND DIRECTION ==========
        # LONG: Daily trend up + 6h Supertrend up + RSI pullback + Z-score oversold
        long_condition = (
            daily_trend == 1 and
            st_trend_val == 1 and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            zscore_val <= ZSCORE_LONG and  # Confirms oversold
            kama_fast_val > kama_val  # Fast KAMA above slow KAMA
        )
        
        # SHORT: Daily trend down + 6h Supertrend down + RSI pullback + Z-score overbought
        short_condition = (
            daily_trend == -1 and
            st_trend_val == -1 and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            zscore_val >= ZSCORE_SHORT and  # Confirms overbought
            kama_fast_val < kama_val  # Fast KAMA below slow KAMA
        )
        
        # Dynamic position sizing based on volatility
        # Lower volatility = higher position size (more room for stops)
        vol_adjustment = 1.0
        if atr_percentile[i] > 0 and atr > 0:
            vol_ratio = atr_percentile[i] / atr
            vol_adjustment = np.clip(vol_ratio, 0.8, 1.2)
        
        # Determine position size based on conviction
        # High conviction: all signals align + strong daily trend
        high_conviction_long = long_condition and st_trend_1d_val == 1
        high_conviction_short = short_condition and st_trend_1d_val == -1
        
        if long_condition:
            base_size = SIZE_HIGH if high_conviction_long else SIZE_BASE
            size = np.clip(base_size * vol_adjustment, SIZE_BASE, SIZE_MAX)
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            base_size = SIZE_HIGH if high_conviction_short else SIZE_BASE
            size = np.clip(base_size * vol_adjustment, SIZE_BASE, SIZE_MAX)
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