#!/usr/bin/env python3
"""
EXPERIMENT #048 - DEMA MACD RSI Pullback (30m Primary + 4h Trend)
==================================================================================================
Hypothesis: Current best uses 1h/4h with KAMA+Supertrend. This uses 30m/4h with DEMA+MACD for
MORE trades (2x vs 1h) while maintaining quality. DEMA responds faster than KAMA/EMA to trend
changes. MACD histogram provides momentum confirmation. 30m should generate 2x more trades than
1h strategies while 4h trend filter eliminates counter-trend noise.

Key innovations:
1. 30m PRIMARY + 4h HTF: 2x more trades than 1h, cleaner than 15m
2. DEMA for fast trend: Double EMA responds faster than KAMA/HMA
3. MACD histogram for momentum: Confirms trend strength before entry
4. RSI pullback entries: Enter on RSI 40-60 in trend direction
5. Position sizing: 0.20 base, 0.30 high conviction (conservative)
6. Stoploss: 2.0*ATR (tighter than 2.5*ATR to reduce drawdown)

Why this should beat kama_supertrend_rsi_pullback_1h_4h_v1 (Sharpe=0.534):
- 30m timeframe = 2x more trading opportunities than 1h
- DEMA faster response = earlier entries in trends
- MACD histogram = additional momentum filter reduces false signals
- Tighter stoploss (2.0*ATR) = lower drawdown
- Conservative sizing (0.20/0.30) = better risk-adjusted returns
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "dema_macd_rsi_pullback_30m_4h_v1"
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


def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    return ema


def calculate_dema(close, period=21):
    """
    Double Exponential Moving Average (DEMA)
    DEMA = 2*EMA1 - EMA2(EMA1)
    Responds faster than EMA with less lag
    """
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    ema1 = calculate_ema(close, period)
    ema2 = calculate_ema(ema1, period)
    
    dema = 2 * ema1 - ema2
    return dema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """
    MACD Indicator
    Returns: macd_line, signal_line, histogram
    """
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
    """Calculate RSI with proper min_periods"""
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
    
    # ========== 30m INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    dema_30m_fast = calculate_dema(close, period=8)
    dema_30m_slow = calculate_dema(close, period=21)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h DEMA for trend direction
        dema_4h_fast = calculate_dema(close_4h, period=8)
        dema_4h_slow = calculate_dema(close_4h, period=21)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        macd_4h_line, macd_4h_signal, macd_4h_hist = calculate_macd(close_4h, fast=12, slow=26, signal=9)
        
        # Align to 30m timeframe (auto shift for completed bars)
        dema_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, dema_4h_slow)
        macd_4h_hist_aligned = align_htf_to_ltf(prices, df_4h, macd_4h_hist)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
    except Exception:
        dema_4h_slow_aligned = np.zeros(n)
        macd_4h_hist_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.20    # Base position (20% of capital)
    SIZE_HIGH = 0.30    # High conviction (30% of capital)
    
    # ATR stoploss - TIGHTER to reduce drawdown
    ATR_STOP_MULT = 2.0
    
    # RSI pullback zones
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # MACD histogram threshold for momentum
    MACD_THRESH = 0.0
    
    first_valid = 100
    
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
        dema_fast = dema_30m_fast[i]
        dema_slow = dema_30m_slow[i]
        macd_histogram = macd_hist[i]
        
        # 4h trend filters (MASTER FILTER)
        dema_4h_slow_val = dema_4h_slow_aligned[i]
        macd_4h_hist_val = macd_4h_hist_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if dema_4h_slow_val > 0 and price > dema_4h_slow_val:
            trend_4h = 1
        elif dema_4h_slow_val > 0 and price < dema_4h_slow_val:
            trend_4h = -1
        
        # MACD 4h confirms trend
        if macd_4h_hist_val > MACD_THRESH:
            trend_4h = max(trend_4h, 1)
        elif macd_4h_hist_val < -MACD_THRESH:
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
                    signals[i] = SIZE_BASE / 2  # Reduce to half of base
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
                    signals[i] = -SIZE_BASE / 2  # Reduce to half of base
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
        # LONG: 4h trend up + 30m DEMA fast > slow + RSI pullback (40-60) + MACD hist > 0
        long_condition = (
            trend_4h == 1 and
            dema_fast > dema_slow and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            macd_histogram > MACD_THRESH
        )
        
        # SHORT: 4h trend down + 30m DEMA fast < slow + RSI pullback (40-60) + MACD hist < 0
        short_condition = (
            trend_4h == -1 and
            dema_fast < dema_slow and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            macd_histogram < -MACD_THRESH
        )
        
        # Determine position size based on conviction
        # High conviction: all signals align + strong 4h MACD
        high_conviction_long = long_condition and macd_4h_hist_val > MACD_THRESH * 2
        high_conviction_short = short_condition and macd_4h_hist_val < -MACD_THRESH * 2
        
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