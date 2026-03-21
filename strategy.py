#!/usr/bin/env python3
"""
EXPERIMENT #041 - DEMA MACD Momentum with Daily Trend Filter (4h Primary)
==================================================================================================
Hypothesis: Current best uses 1h+4h with KAMA+StochRSI. This uses 4h+1d with DEMA+MACD.

Key innovations:
1. 4h PRIMARY + 1d HTF: Cleaner signals than 1h, more trades than daily-only
2. DEMA for trend: Double EMA reduces lag vs single EMA, more responsive than HMA
3. MACD histogram for momentum: Confirms trend strength before entry
4. Daily trend as master filter: Only trade in direction of 1d trend (highest conviction)
5. Conservative sizing: 0.25 base, 0.35 high conviction, 2.5 ATR stoploss

Why this should beat hma_rsi_pullback_daily_trend_4h_v1 (Sharpe=0.537):
- 4h timeframe captures major moves without 1h noise
- DEMA has less lag than HMA for faster trend detection
- MACD histogram provides momentum confirmation RSI lacks
- 1d trend filter is stronger than 4h trend filter
- Fewer trades but higher quality = better Sharpe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "dema_macd_momentum_4h_daily_v1"
timeframe = "4h"
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


def calculate_dema(close, period=21):
    """
    Double Exponential Moving Average (DEMA)
    DEMA = 2*EMA - EMA(EMA)
    Reduces lag compared to single EMA
    """
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    
    dema = (2 * ema1 - ema2).values
    dema[:period * 2 - 1] = 0  # Mark invalid periods
    return dema


def calculate_macd(close, fast_period=12, slow_period=26, signal_period=9):
    """
    MACD Indicator
    Returns: macd_line, signal_line, histogram
    """
    n = len(close)
    if n < slow_period + signal_period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    ema_slow = close_series.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    
    macd_line = (ema_fast - ema_slow).values
    macd_series = pd.Series(macd_line)
    signal_line = macd_series.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean().values
    histogram = macd_line - signal_line
    
    # Mark invalid periods
    invalid = slow_period + signal_period - 1
    macd_line[:invalid] = 0
    signal_line[:invalid] = 0
    histogram[:invalid] = 0
    
    return macd_line, signal_line, histogram


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 4h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_4h = calculate_atr(high, low, close, period=14)
    dema_4h = calculate_dema(close, period=21)
    dema_fast_4h = calculate_dema(close, period=10)
    macd_4h, macd_signal_4h, macd_hist_4h = calculate_macd(close, fast_period=12, slow_period=26, signal_period=9)
    supertrend_4h, st_trend_4h = calculate_supertrend(high, low, close, atr_4h, multiplier=3.0)
    rsi_4h = calculate_rsi(close, period=14)
    
    # ========== 1d INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # 1d DEMA for master trend direction
        dema_1d = calculate_dema(close_1d, period=21)
        atr_1d = calculate_atr(high_1d, low_1d, close_1d, period=14)
        _, st_trend_1d = calculate_supertrend(high_1d, low_1d, close_1d, atr_1d, multiplier=3.0)
        macd_1d, _, macd_hist_1d = calculate_macd(close_1d, fast_period=12, slow_period=26, signal_period=9)
        
        # Align to 4h timeframe (auto shift for completed bars)
        dema_1d_aligned = align_htf_to_ltf(prices, df_1d, dema_1d)
        st_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, st_trend_1d)
        macd_hist_1d_aligned = align_htf_to_ltf(prices, df_1d, macd_hist_1d)
        
    except Exception:
        dema_1d_aligned = np.zeros(n)
        st_trend_1d_aligned = np.zeros(n)
        macd_hist_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.25    # Base position (25% of capital)
    SIZE_HIGH = 0.35    # High conviction (35% of capital)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.5
    
    # MACD histogram threshold for momentum confirmation
    MACD_HIST_MIN = 0  # Must be positive for long, negative for short
    
    # RSI filter zones
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 70
    RSI_SHORT_MIN = 30
    RSI_SHORT_MAX = 60
    
    first_valid = 200  # Need enough data for 1d alignment
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_4h[i]) or atr_4h[i] == 0 or np.isnan(dema_4h[i]) or dema_4h[i] == 0:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        dema_val = dema_4h[i]
        dema_fast_val = dema_fast_4h[i]
        macd_hist = macd_hist_4h[i]
        st_trend_val = st_trend_4h[i]
        rsi_val = rsi_4h[i]
        
        # 1d trend filters (MASTER FILTER)
        dema_1d_val = dema_1d_aligned[i]
        st_trend_1d_val = st_trend_1d_aligned[i]
        macd_hist_1d = macd_hist_1d_aligned[i]
        
        # Determine 1d trend direction
        one_d_trend = 0
        if dema_1d_val > 0 and price > dema_1d_val:
            one_d_trend = 1
        elif dema_1d_val > 0 and price < dema_1d_val:
            one_d_trend = -1
        
        if st_trend_1d_val == 1:
            one_d_trend = max(one_d_trend, 1)
        elif st_trend_1d_val == -1:
            one_d_trend = min(one_d_trend, -1)
        
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
        
        # ========== ENTRY LOGIC - MACD MOMENTUM IN DAILY TREND DIRECTION ==========
        # LONG: 1d trend up + 4h Supertrend up + MACD histogram positive + DEMA aligned
        long_condition = (
            one_d_trend == 1 and
            st_trend_val == 1 and
            macd_hist > MACD_HIST_MIN and
            dema_fast_val > dema_val and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX
        )
        
        # SHORT: 1d trend down + 4h Supertrend down + MACD histogram negative + DEMA aligned
        short_condition = (
            one_d_trend == -1 and
            st_trend_val == -1 and
            macd_hist < -MACD_HIST_MIN and
            dema_fast_val < dema_val and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX
        )
        
        # High conviction: 1d MACD also confirms + 1d Supertrend confirms
        high_conviction_long = (
            long_condition and 
            st_trend_1d_val == 1 and 
            macd_hist_1d > MACD_HIST_MIN
        )
        high_conviction_short = (
            short_condition and 
            st_trend_1d_val == -1 and 
            macd_hist_1d < -MACD_HIST_MIN
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