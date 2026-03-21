#!/usr/bin/env python3
"""
EXPERIMENT #024 - HMA BB Regime MTF 4h+1d
==================================================================================================
Hypothesis: Current best (hma_rsi_pullback_daily_trend_4h_v1, Sharpe=0.537) uses 4h primary + 1d filter.
This replicates that successful timeframe combo but adds Bollinger Band regime detection to avoid
choppy markets. HMA is proven superior to KAMA/DEMA for trend following. RSI pullback entries work
well in trending markets. BB squeeze detection filters low-volatility chop where trend strategies fail.

Key innovations:
1. 4h PRIMARY + 1d HTF: Same as current best (proven combination)
2. HMA(21) trend: Faster than EMA, smoother than SMA (proven in best strategy)
3. Bollinger Band regime: Only trade when BB width > 20th percentile (avoid squeeze/chop)
4. RSI(14) pullback: Enter on 45-55 RSI in direction of trend (not extreme levels)
5. 2.0*ATR stoploss: Balanced between #023's 1.5*ATR (too tight) and loose stops
6. Discrete sizing: 0.25 base, 0.35 high conviction (minimize fee churn)

Why this should beat #023 (Sharpe=0.336) and approach best (Sharpe=0.537):
- 4h+1d MTF combo is proven (current best uses this)
- HMA is proven superior to KAMA for trend following
- BB regime filter avoids choppy periods where trend strategies lose
- RSI pullback (45-55) catches continuations, not reversals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_bb_regime_mtf_4h_1d_v1"
timeframe = "4h"
leverage = 1.0


def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Eliminates lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    # Helper function for WMA
    def wma(data, w_period):
        result = np.zeros(len(data))
        weights = np.arange(1, w_period + 1)
        for i in range(w_period - 1, len(data)):
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    
    # WMA of diff with sqrt(n) period
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """
    Bollinger Bands
    Returns: upper, middle, lower, bandwidth, percent_b
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    # Bandwidth = (Upper - Lower) / Middle
    bandwidth = np.zeros(n)
    mask = middle > 0
    bandwidth[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
    # Percent B = (Close - Lower) / (Upper - Lower)
    percent_b = np.zeros(n)
    mask2 = (upper - lower) > 0
    percent_b[mask2] = (close[mask2] - lower[mask2]) / (upper[mask2] - lower[mask2])
    
    return upper, middle, lower, bandwidth, percent_b


def calculate_bb_width_percentile(bandwidth, lookback=100):
    """
    Calculate rolling percentile of BB width to detect squeeze/expansion
    Returns percentile rank (0-100)
    """
    n = len(bandwidth)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bandwidth[i - lookback + 1:i + 1]
        valid = window[window > 0]
        if len(valid) > 0:
            # Calculate percentile rank of current bandwidth
            rank = np.sum(valid <= bandwidth[i]) / len(valid) * 100
            percentile[i] = rank
        else:
            percentile[i] = 50
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 4h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_4h = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=10)
    
    # Bollinger Bands for regime detection
    bb_upper, bb_middle, bb_lower, bb_width, bb_pct = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_percentile = calculate_bb_width_percentile(bb_width, lookback=100)
    
    # ========== 1d INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # 1d HMA for primary trend direction
        hma_1d = calculate_hma(close_1d, period=21)
        rsi_1d = calculate_rsi(close_1d, period=14)
        atr_1d = calculate_atr(high_1d, low_1d, close_1d, period=14)
        
        # Align to 4h timeframe (auto shift for completed bars)
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
        rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
        
    except Exception:
        hma_1d_aligned = np.zeros(n)
        rsi_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE & DISCRETE
    SIZE_BASE = 0.25    # Base position (25% of capital)
    SIZE_HIGH = 0.35    # High conviction (35% of capital)
    
    # ATR stoploss - BALANCED (not too tight like #023's 1.5*ATR)
    ATR_STOP_MULT = 2.0
    
    # RSI pullback zones (neutral zone, not extremes)
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 58
    RSI_SHORT_MIN = 42
    RSI_SHORT_MAX = 55
    
    # BB regime filter - only trade when not in squeeze
    BB_MIN_PERCENTILE = 20  # Only trade when BB width > 20th percentile
    
    first_valid = max(150, 100)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_4h[i]) or atr_4h[i] == 0 or np.isnan(rsi_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        rsi_val = rsi_4h[i]
        hma_val = hma_4h[i]
        hma_fast_val = hma_4h_fast[i]
        bb_width_pct = bb_percentile[i]
        
        # 1d trend filters (MASTER FILTER)
        hma_1d_val = hma_1d_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        
        # Determine 1d trend direction
        trend_1d = 0
        if hma_1d_val > 0:
            if price > hma_1d_val:
                trend_1d = 1
            elif price < hma_1d_val:
                trend_1d = -1
        
        # RSI confirmation on 1d
        if rsi_1d_val > 55:
            trend_1d = max(trend_1d, 1)
        elif rsi_1d_val < 45:
            trend_1d = min(trend_1d, -1)
        
        # BB regime filter - skip if in squeeze (low volatility = chop)
        in_expansion = bb_width_pct >= BB_MIN_PERCENTILE
        
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
        
        # ========== ENTRY LOGIC - HMA + RSI + BB REGIME ==========
        # Skip entries if in BB squeeze (choppy market)
        if not in_expansion:
            signals[i] = 0.0
            continue
        
        # LONG: 1d trend up + 4h HMA up + HMA fast > slow + RSI pullback + BB expansion
        long_condition = (
            trend_1d == 1 and
            hma_fast_val > hma_val and
            price > hma_val and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX
        )
        
        # SHORT: 1d trend down + 4h HMA down + HMA fast < slow + RSI pullback + BB expansion
        short_condition = (
            trend_1d == -1 and
            hma_fast_val < hma_val and
            price < hma_val and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX
        )
        
        # Determine position size based on conviction
        # High conviction: 1d RSI confirms + strong HMA separation
        hma_separation = abs(hma_fast_val - hma_val) / hma_val if hma_val > 0 else 0
        high_conviction_long = long_condition and rsi_1d_val > 55 and hma_separation > 0.005
        high_conviction_short = short_condition and rsi_1d_val < 45 and hma_separation > 0.005
        
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