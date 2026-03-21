#!/usr/bin/env python3
"""
EXPERIMENT #018 - MACD RSI Volume Momentum with 4h Trend Filter (30m Primary)
==================================================================================================
Hypothesis: Current best uses 4h primary + 1d filter. This uses 30m primary + 4h filter for more
trades while maintaining quality. MACD momentum + RSI pullback + volume confirmation should catch
trend continuations with better timing than HMA alone.

Key innovations:
1. 30m PRIMARY + 4h HTF: More trades than 4h primary, less noise than 15m strategies
2. MACD histogram for momentum direction (proven in exp#007 Sharpe=0.488)
3. RSI pullback zones (40-60) for entry timing in trend direction
4. Volume spike filter (1.5x average) to confirm genuine breakouts
5. 2.5 ATR stoploss (wider than 2.0 to reduce whipsaw exits on 30m)
6. Discrete sizing: 0.25 base, 0.35 high conviction

Why this should beat hma_rsi_pullback_daily_trend_4h_v1 (Sharpe=0.537):
- 30m captures more intraday momentum moves than 4h
- MACD + RSI combo proven in exp#007 (Sharpe=0.488)
- Volume filter reduces false breakouts
- More trades = better statistical significance
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "macd_rsi_volume_momentum_30m_4h_v1"
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


def calculate_ema(series, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
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


def calculate_sma(series, period):
    """Calculate Simple Moving Average"""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values


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
    volume = prices["volume"].values
    n = len(close)
    
    # ========== 30m INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    supertrend_30m, st_trend_30m = calculate_supertrend(high, low, close, atr_30m, multiplier=3.0)
    
    # Volume SMA for spike detection
    vol_sma_30m = calculate_sma(volume, 20)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        volume_4h = df_4h['volume'].values
        
        # 4h trend indicators
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        macd_4h_line, macd_4h_signal, macd_4h_hist = calculate_macd(close_4h, fast=12, slow=26, signal=9)
        
        # 4h EMA for trend direction
        ema_4h_21 = calculate_ema(close_4h, 21)
        ema_4h_50 = calculate_ema(close_4h, 50)
        
        # Align to 30m timeframe (auto shift for completed bars)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        macd_4h_hist_aligned = align_htf_to_ltf(prices, df_4h, macd_4h_hist)
        ema_4h_21_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_21)
        ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
        
    except Exception:
        st_trend_4h_aligned = np.zeros(n)
        macd_4h_hist_aligned = np.zeros(n)
        ema_4h_21_aligned = np.zeros(n)
        ema_4h_50_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.25   # Base position (25% of capital)
    SIZE_HIGH = 0.35   # High conviction (35% of capital)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.5  # Wider stop for 30m to reduce whipsaws
    
    # RSI pullback zones
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Volume spike threshold
    VOL_SPIKE_MULT = 1.5
    
    first_valid = max(100, 60)
    
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
        st_trend_val = st_trend_30m[i]
        macd_h = macd_hist[i]
        macd_h_prev = macd_hist[i - 1] if i > 0 else 0
        
        # 4h trend filters (MASTER FILTER)
        st_trend_4h_val = st_trend_4h_aligned[i]
        macd_4h_h = macd_4h_hist_aligned[i]
        ema_4h_21_val = ema_4h_21_aligned[i]
        ema_4h_50_val = ema_4h_50_aligned[i]
        
        # Volume spike detection
        vol_avg = vol_sma_30m[i]
        vol_spike = volume[i] > (vol_avg * VOL_SPIKE_MULT) if vol_avg > 0 else False
        
        # Determine 4h trend direction
        trend_4h = 0
        if ema_4h_21_val > 0 and ema_4h_50_val > 0:
            if ema_4h_21_val > ema_4h_50_val:
                trend_4h = 1
            elif ema_4h_21_val < ema_4h_50_val:
                trend_4h = -1
        
        if st_trend_4h_val == 1:
            trend_4h = max(trend_4h, 1)
        elif st_trend_4h_val == -1:
            trend_4h = min(trend_4h, -1)
        
        if macd_4h_h > 0:
            trend_4h = max(trend_4h, 1)
        elif macd_4h_h < 0:
            trend_4h = min(trend_4h, -1)
        
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
        
        # ========== ENTRY LOGIC - MACD + RSI + VOLUME IN TREND DIRECTION ==========
        # MACD momentum confirmation
        macd_bullish = macd_h > 0 and macd_h > macd_h_prev  # Positive and rising
        macd_bearish = macd_h < 0 and macd_h < macd_h_prev  # Negative and falling
        
        # LONG: 4h trend up + 30m Supertrend up + MACD bullish + RSI pullback + Volume spike
        long_condition = (
            trend_4h == 1 and
            st_trend_val == 1 and
            macd_bullish and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX
        )
        
        # SHORT: 4h trend down + 30m Supertrend down + MACD bearish + RSI pullback + Volume spike
        short_condition = (
            trend_4h == -1 and
            st_trend_val == -1 and
            macd_bearish and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX
        )
        
        # Determine position size based on conviction
        # High conviction: all signals align + volume spike confirmation
        high_conviction_long = long_condition and vol_spike and macd_4h_h > 0
        high_conviction_short = short_condition and vol_spike and macd_4h_h < 0
        
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