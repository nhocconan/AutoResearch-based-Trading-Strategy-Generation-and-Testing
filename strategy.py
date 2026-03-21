#!/usr/bin/env python3
"""
EXPERIMENT #033 - HMA Trend + RSI Pullback (30m Primary, 4h Trend Filter)
==================================================================================================
Hypothesis: Current best (hma_rsi_pullback_daily_trend_4h_v1, Sharpe=0.537) uses 4h primary with daily filter.
This uses 30m primary with 4h trend filter for MORE trade opportunities while maintaining signal quality.
30m captures intraday swings that 4h misses, while 4h HMA filter prevents counter-trend trades.

Key changes vs #032 (keltner_squeeze_breakout_4h_trend_1h_v1, Sharpe=0.168):
1. 30m PRIMARY instead of 1h: More trades, captures intraday momentum swings
2. HMA trend + RSI pullback instead of Keltner squeeze: Proven combination (best strategy uses this)
3. Simpler entry logic: RSI pullback to 40-60 range in trend direction (no complex squeeze detection)
4. Tighter stoploss: 1.5*ATR instead of 2.0*ATR (reduces drawdown per trade)
5. Position sizing: 0.30 base, 0.35 high conviction (discrete levels to reduce fee churn)

Why this should beat Sharpe=0.537:
- 30m timeframe captures 3-5x more trades than 4h while avoiding 15m noise
- HMA is faster than SMA for trend detection (less lag in crypto)
- RSI pullback entries have better risk/reward than breakout entries
- 4h trend filter is strong enough to avoid counter-trend traps
- Tighter stops reduce per-trade loss, improving Sharpe ratio
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_pullback_30m_4h_v1"
timeframe = "30m"
leverage = 1.0


def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        result = np.zeros(len(series))
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = 2 * wma_half - wma_full
    
    hma = wma(diff, sqrt_period)
    
    return hma


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


def calculate_hma_trend_direction(hma_values, lookback=3):
    """
    Determine trend direction from HMA slope
    Returns: 1 = uptrend, -1 = downtrend, 0 = neutral
    """
    n = len(hma_values)
    trend = np.zeros(n)
    
    for i in range(lookback, n):
        if hma_values[i] > hma_values[i - lookback]:
            trend[i] = 1
        elif hma_values[i] < hma_values[i - lookback]:
            trend[i] = -1
        else:
            trend[i] = 0
    
    return trend


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 30m INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    
    # HMA for trend on 30m
    hma_30m_short = calculate_hma(close, period=16)
    hma_30m_long = calculate_hma(close, period=48)
    hma_trend_30m = calculate_hma_trend_direction(hma_30m_long, lookback=3)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h HMA for master trend direction
        hma_4h = calculate_hma(close_4h, period=21)
        hma_trend_4h = calculate_hma_trend_direction(hma_4h, lookback=3)
        
        # 4h RSI for momentum confirmation
        rsi_4h = calculate_rsi(close_4h, period=14)
        
        # 4h ATR for volatility adjustment
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        
        # Align to 30m timeframe (auto shift for completed bars)
        hma_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_trend_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        
    except Exception:
        hma_trend_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
        hma_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to minimize fee churn
    SIZE_BASE = 0.30    # Base position (30% of capital)
    SIZE_HIGH = 0.35    # High conviction (35% of capital)
    SIZE_MAX = 0.40     # Absolute maximum
    
    # ATR stoploss - TIGHTER than previous (1.5 instead of 2.0)
    ATR_STOP_MULT = 1.5
    
    # RSI pullback thresholds
    RSI_LONG_ENTRY = 45   # Enter long on pullback to 45 in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on pullback to 55 in downtrend
    RSI_EXIT = 65         # Exit long when RSI reaches 65 (overbought)
    RSI_EXIT_SHORT = 35   # Exit short when RSI reaches 35 (oversold)
    
    first_valid = max(100, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    entry_rsi = np.zeros(n)
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
        hma_trend_val = hma_trend_30m[i]
        
        # 4h trend filters (MASTER FILTER - must align with 4h trend)
        hma_trend_4h_val = hma_trend_4h_aligned[i]
        rsi_4h_val = rsi_4h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        hma_4h_val = hma_4h_aligned[i]
        
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
            
            # Stoploss check (1.5*ATR - tighter)
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
                
                # RSI exit (overbought)
                if rsi_val >= RSI_EXIT:
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
                
                # 4h trend reversal exit
                if hma_trend_4h_val == -1:
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
                
                # RSI exit (oversold)
                if rsi_val <= RSI_EXIT_SHORT:
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
                
                # 4h trend reversal exit
                if hma_trend_4h_val == 1:
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
        # LONG: 4h uptrend + 30m pullback (RSI < 50) + RSI crossing back up
        long_trend_filter = hma_trend_4h_val == 1
        long_pullback = rsi_val < 50 and rsi_val >= RSI_LONG_ENTRY
        long_momentum = rsi_val > entry_rsi[i - 1] if i > 0 else True  # RSI rising
        long_30m_trend = hma_trend_val >= 0  # 30m not in downtrend
        
        long_condition = (
            long_trend_filter and
            long_pullback and
            long_30m_trend
        )
        
        # SHORT: 4h downtrend + 30m pullback (RSI > 50) + RSI crossing back down
        short_trend_filter = hma_trend_4h_val == -1
        short_pullback = rsi_val > 50 and rsi_val <= RSI_SHORT_ENTRY
        short_momentum = rsi_val < entry_rsi[i - 1] if i > 0 else True  # RSI falling
        short_30m_trend = hma_trend_val <= 0  # 30m not in uptrend
        
        short_condition = (
            short_trend_filter and
            short_pullback and
            short_30m_trend
        )
        
        # Determine position size based on conviction
        # High conviction: strong 4h trend + RSI deep in pullback zone
        high_conviction_long = long_condition and rsi_4h_val > 55 and rsi_val <= 40
        high_conviction_short = short_condition and rsi_4h_val < 45 and rsi_val >= 60
        
        # Dynamic sizing: reduce size in high volatility (high 4h ATR)
        vol_adjustment = 1.0
        if atr_4h_val > 0 and price > 0:
            atr_pct = (atr_4h_val / price) * 100
            if atr_pct > 4.0:
                vol_adjustment = 0.85
            elif atr_pct > 6.0:
                vol_adjustment = 0.70
        
        if long_condition:
            base_size = SIZE_HIGH if high_conviction_long else SIZE_BASE
            size = min(base_size * vol_adjustment, SIZE_MAX)
            # Discretize to 0.05 increments
            size = round(size * 20) / 20
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            entry_rsi[i] = rsi_val
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            base_size = SIZE_HIGH if high_conviction_short else SIZE_BASE
            size = min(base_size * vol_adjustment, SIZE_MAX)
            # Discretize to 0.05 increments
            size = round(size * 20) / 20
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            entry_rsi[i] = rsi_val
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Track entry RSI for momentum check
        if position_side[i] == 0:
            entry_rsi[i] = entry_rsi[i - 1] if i > 0 else 50
    
    return signals