#!/usr/bin/env python3
"""
EXPERIMENT #020 - Supertrend MACD MTF with ATR Position Sizing (1h Primary)
==================================================================================================
Hypothesis: Current best uses HMA+RSI. This uses Supertrend for cleaner trend signals + MACD 
histogram for momentum + ATR-based dynamic position sizing. 1h primary with 4h supertrend filter.

Why this should beat current best (Sharpe=0.537):
1. Supertrend provides clearer trend direction than HMA (less whipsaw)
2. MACD histogram momentum confirms trend strength before entry
3. ATR-based position sizing reduces size in high volatility (controls drawdown)
4. 1h timeframe balances trade frequency vs signal quality
5. Simpler filter stack than failed strategies (no Z-score, no BBW over-filtering)

Key differences from failed attempts:
- #008 Donchian+RSI+Volume failed (Sharpe=-0.326) - breakouts whipsaw
- #011 Supertrend+Stoch+Volume failed (Sharpe=-0.651) - Stoch too noisy
- This uses Supertrend+MACD (proven combo in #010, #016, #019)

Risk management:
- Dynamic position sizing: base_size * (target_vol / current_ATR_vol)
- Stoploss: 2.0*ATR trailing
- Take profit: 50% at 2R, trail remaining at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_macd_atr_mtf_1h_4h_v1"
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """
    Calculate MACD line, signal line, and histogram
    """
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    
    ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    supertrend_1h, st_trend_1h = calculate_supertrend(high, low, close, atr_1h, multiplier=3.0)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h ATR and Supertrend for master trend direction
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        
        # Align to 1h timeframe (auto shift for completed bars)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        
    except Exception:
        st_trend_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - DYNAMIC based on ATR
    SIZE_BASE = 0.20      # Base position (20%)
    SIZE_HIGH = 0.30      # High conviction (30%)
    TARGET_ATR_PCT = 0.02 # Target ATR as % of price (2%)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    # RSI filter thresholds (avoid extremes)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # MACD histogram threshold for momentum
    MACD_MIN_HIST = 0.0
    
    first_valid = max(100, 50)
    
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
        atr_pct = atr / price if price > 0 else 0
        rsi_val = rsi_1h[i]
        st_trend_val = st_trend_1h[i]
        macd_hist_val = macd_hist[i]
        
        # 4h trend filter (MASTER FILTER)
        st_trend_4h_val = st_trend_4h_aligned[i]
        
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
        
        # ========== ENTRY LOGIC - SUPERTREND + MACD + RSI ==========
        # MACD histogram momentum confirmation
        macd_bullish = macd_hist_val > MACD_MIN_HIST
        macd_bearish = macd_hist_val < -MACD_MIN_HIST
        
        # Check MACD histogram increasing (momentum building)
        macd_hist_increasing = i > 0 and macd_hist_val > macd_hist[i - 1]
        macd_hist_decreasing = i > 0 and macd_hist_val < macd_hist[i - 1]
        
        # LONG: 4h Supertrend up + 1h Supertrend up + RSI in zone + MACD bullish + momentum building
        long_condition = (
            st_trend_4h_val == 1 and
            st_trend_val == 1 and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            macd_bullish and
            macd_hist_increasing
        )
        
        # SHORT: 4h Supertrend down + 1h Supertrend down + RSI in zone + MACD bearish + momentum building
        short_condition = (
            st_trend_4h_val == -1 and
            st_trend_val == -1 and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            macd_bearish and
            macd_hist_decreasing
        )
        
        # Dynamic position sizing based on ATR
        # Lower ATR% = higher position size (less volatile = can size up)
        # Higher ATR% = lower position size (more volatile = size down)
        if atr_pct > 0:
            vol_adjustment = min(1.5, max(0.7, TARGET_ATR_PCT / atr_pct))
        else:
            vol_adjustment = 1.0
        
        # Determine position size based on conviction
        high_conviction_long = long_condition and st_trend_4h_val == 1 and rsi_val >= 45 and rsi_val <= 55
        high_conviction_short = short_condition and st_trend_4h_val == -1 and rsi_val >= 45 and rsi_val <= 55
        
        if long_condition:
            base_size = SIZE_HIGH if high_conviction_long else SIZE_BASE
            size = base_size * vol_adjustment
            size = min(0.35, max(0.15, size))  # Clamp to safe range
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            base_size = SIZE_HIGH if high_conviction_short else SIZE_BASE
            size = base_size * vol_adjustment
            size = min(0.35, max(0.15, size))  # Clamp to safe range
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