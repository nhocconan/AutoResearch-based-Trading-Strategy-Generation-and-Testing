#!/usr/bin/env python3
"""
EXPERIMENT #026 - KAMA Adaptive Trend with Volume Confirmation (1h Primary)
==================================================================================================
Hypothesis: Current best uses HMA on 4h. This uses KAMA (Kaufman Adaptive MA) on 1h for faster
adaptation to volatility changes. KAMA flattens in choppy markets and trends in directional moves.
Adding volume confirmation (taker_buy_ratio) filters false breakouts. 4h trend filter keeps us
on the right side of major moves.

Key innovations vs current best (hma_rsi_pullback_daily_trend_4h_v1):
1. KAMA instead of HMA - adapts efficiency ratio to market noise (better in ranging markets)
2. 1h primary vs 4h - more trades (50-200 vs 20-50), faster entry/exit
3. Volume confirmation - taker_buy_ratio > 0.55 for longs, < 0.45 for shorts
4. 4h KAMA trend filter (not daily) - more responsive than daily, cleaner than 1h
5. Same conservative sizing: 0.25 base, 0.35 high conviction, 2.0 ATR stop

Why this should beat Sharpe=0.537:
- KAMA's adaptive nature reduces whipsaws in choppy markets (major drawdown source)
- 1h timeframe captures more moves while 4h filter prevents counter-trend trades
- Volume filter eliminates low-conviction breakouts that reverse quickly
- More trades = better statistical significance while keeping DD controlled
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_volume_trend_mtf_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise using Efficiency Ratio (ER)
    ER = |net change| / sum of absolute changes over n periods
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = prior_KAMA + SC * (price - prior_KAMA)
    """
    n = len(close)
    if n < efficiency_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(efficiency_period, n):
        net_change = abs(close[i] - close[i - efficiency_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - efficiency_period:i + 1])))
        if sum_changes > 0:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA at SMA of first slow_period values
    kama[slow_period - 1] = np.mean(close[:slow_period])
    
    # Calculate KAMA
    for i in range(slow_period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_taker_buy_ratio(prices):
    """Calculate taker buy volume ratio (volume confirmation)"""
    n = len(prices)
    ratio = np.zeros(n)
    
    volume = prices['volume'].values
    taker_buy = prices['taker_buy_volume'].values
    
    for i in range(n):
        if volume[i] > 0:
            ratio[i] = taker_buy[i] / volume[i]
        else:
            ratio[i] = 0.5
    
    return ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    kama_1h = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    kama_1h_fast = calculate_kama(close, efficiency_period=5, fast_period=2, slow_period=15)
    taker_ratio = calculate_taker_buy_ratio(prices)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h KAMA for trend direction
        kama_4h = calculate_kama(close_4h, efficiency_period=10, fast_period=2, slow_period=30)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        
        # Align to 1h timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.25    # Base position (25% of capital)
    SIZE_HIGH = 0.35    # High conviction (35% of capital)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    # RSI entry zones (pullback in trend)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Volume confirmation thresholds
    VOLUME_LONG_MIN = 0.52   # More buyers than sellers
    VOLUME_SHORT_MAX = 0.48  # More sellers than buyers
    
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
        rsi_val = rsi_1h[i]
        kama_val = kama_1h[i]
        kama_fast_val = kama_1h_fast[i]
        vol_ratio = taker_ratio[i]
        
        # 4h trend filters (MASTER FILTER)
        kama_4h_val = kama_4h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if kama_4h_val > 0 and price > kama_4h_val:
            trend_4h = 1
        elif kama_4h_val > 0 and price < kama_4h_val:
            trend_4h = -1
        
        # KAMA fast/slow cross on 1h
        kama_cross_up = kama_fast_val > kama_val and kama_fast_val > 0 and kama_val > 0
        kama_cross_down = kama_fast_val < kama_val and kama_fast_val > 0 and kama_val > 0
        
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
        
        # ========== ENTRY LOGIC - KAMA TREND + VOLUME CONFIRMATION ==========
        # LONG: 4h trend up + 1h KAMA cross up + RSI pullback + volume confirmation
        long_condition = (
            trend_4h == 1 and
            kama_cross_up and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            vol_ratio >= VOLUME_LONG_MIN
        )
        
        # SHORT: 4h trend down + 1h KAMA cross down + RSI pullback + volume confirmation
        short_condition = (
            trend_4h == -1 and
            kama_cross_down and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            vol_ratio <= VOLUME_SHORT_MAX
        )
        
        # High conviction: strong 4h trend (price far from 4h KAMA)
        high_conviction_long = False
        high_conviction_short = False
        
        if kama_4h_val > 0 and long_condition:
            distance_pct = (price - kama_4h_val) / kama_4h_val
            if distance_pct > 0.01:  # Price > 1% above 4h KAMA
                high_conviction_long = True
        
        if kama_4h_val > 0 and short_condition:
            distance_pct = (kama_4h_val - price) / kama_4h_val
            if distance_pct > 0.01:  # Price > 1% below 4h KAMA
                high_conviction_short = True
        
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