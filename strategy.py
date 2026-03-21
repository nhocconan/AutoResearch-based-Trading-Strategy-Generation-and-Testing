#!/usr/bin/env python3
"""
EXPERIMENT #031 - HMA ADX RSI Pullback with Daily Trend Filter (4h Primary)
==================================================================================================
Hypothesis: Current best uses 4h+1d with HMA+RSI. This improves on it by:
1. Adding ADX filter to only trade when trend strength is high (>25)
2. Using 4h as PRIMARY (not 1h) for cleaner signals, fewer whipsaws
3. 1d HMA as master trend filter (same as current best)
4. Tighter stoploss at 1.5*ATR with trailing at 1R
5. Dynamic position sizing based on ADX strength (higher ADX = larger size)

Why this should beat Sharpe=0.537:
- ADX filter eliminates weak trend periods where mean-reversion dominates
- 4h timeframe has fewer false signals than 1h strategies
- Dynamic sizing based on trend strength improves risk-adjusted returns
- Tighter stops protect capital during reversals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_adx_rsi_pullback_4h_1d_v1"
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


def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    def wma(series, span):
        """Weighted Moving Average"""
        weights = np.arange(1, span + 1)
        result = np.zeros(len(series))
        for i in range(span - 1, len(series)):
            result[i] = np.sum(series[i - span + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    diff = 2 * wma_half - wma_full
    
    hma = wma(diff, sqrt_period)
    return hma


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


def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = strong trend, ADX < 20 = ranging market
    """
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_diff = high[i] - high[i - 1]
        minus_diff = low[i - 1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        elif minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    # Smooth DM and TR
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    plus_dm_smooth[period - 1] = np.sum(plus_dm[1:period])
    minus_dm_smooth[period - 1] = np.sum(minus_dm[1:period])
    
    for i in range(period, n):
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - plus_dm_smooth[i - 1] / period + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - minus_dm_smooth[i - 1] / period + minus_dm[i]
    
    mask_atr = atr > 0
    plus_di[mask_atr] = 100 * plus_dm_smooth[mask_atr] / atr[mask_atr]
    minus_di[mask_atr] = 100 * minus_dm_smooth[mask_atr] / atr[mask_atr]
    
    # DX and ADX
    dx = np.zeros(n)
    mask_di = (plus_di + minus_di) > 0
    dx[mask_di] = 100 * np.abs(plus_di[mask_di] - minus_di[mask_di]) / (plus_di[mask_di] + minus_di[mask_di])
    
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


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
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(high, low, close, period=14)
    
    # ========== 1d INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # 1d HMA for master trend
        hma_1d = calculate_hma(close_1d, period=21)
        
        # Align to 4h timeframe (auto shift for completed bars)
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
        
    except Exception:
        hma_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE with ADX-based adjustment
    SIZE_BASE = 0.20    # Base position (low ADX)
    SIZE_HIGH = 0.30    # High conviction (high ADX)
    SIZE_MAX = 0.35     # Maximum position
    
    # ATR stoploss - tighter than standard
    ATR_STOP_MULT = 1.5
    
    # RSI pullback zones
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ADX thresholds
    ADX_MIN = 20   # Minimum for any trade
    ADX_HIGH = 30  # High conviction threshold
    
    first_valid = max(100, 50)
    
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
        adx_val = adx_4h[i]
        hma_val = hma_4h[i]
        hma_fast_val = hma_4h_fast[i]
        
        # 1d trend filter (MASTER FILTER)
        hma_1d_val = hma_1d_aligned[i]
        
        # Determine 1d trend direction
        trend_1d = 0
        if hma_1d_val > 0:
            if price > hma_1d_val:
                trend_1d = 1
            elif price < hma_1d_val:
                trend_1d = -1
        
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
            
            # Stoploss check (1.5*ATR)
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
        
        # ========== ENTRY LOGIC - RSI PULLBACK IN TREND DIRECTION ==========
        # Skip if ADX too low (ranging market)
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            continue
        
        # Calculate dynamic position size based on ADX strength
        adx_factor = min((adx_val - ADX_MIN) / (ADX_HIGH - ADX_MIN), 1.0) if adx_val > ADX_MIN else 0
        adx_factor = max(0, adx_factor)
        
        # LONG: 1d trend up + 4h HMA up + ADX strong + RSI pullback (35-55)
        long_condition = (
            trend_1d == 1 and
            hma_fast_val > hma_val and
            adx_val >= ADX_MIN and
            plus_di_4h[i] > minus_di_4h[i] and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX
        )
        
        # SHORT: 1d trend down + 4h HMA down + ADX strong + RSI pullback (45-65)
        short_condition = (
            trend_1d == -1 and
            hma_fast_val < hma_val and
            adx_val >= ADX_MIN and
            minus_di_4h[i] > plus_di_4h[i] and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX
        )
        
        # Determine position size based on ADX conviction
        if long_condition:
            base_size = SIZE_HIGH if adx_val >= ADX_HIGH else SIZE_BASE
            size = min(base_size + (SIZE_MAX - SIZE_HIGH) * adx_factor, SIZE_MAX)
            size = round(size * 2) / 2  # Discrete levels: 0.20, 0.25, 0.30, 0.35
            size = max(SIZE_BASE, min(size, SIZE_MAX))
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            base_size = SIZE_HIGH if adx_val >= ADX_HIGH else SIZE_BASE
            size = min(base_size + (SIZE_MAX - SIZE_HIGH) * adx_factor, SIZE_MAX)
            size = round(size * 2) / 2  # Discrete levels
            size = max(SIZE_BASE, min(size, SIZE_MAX))
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