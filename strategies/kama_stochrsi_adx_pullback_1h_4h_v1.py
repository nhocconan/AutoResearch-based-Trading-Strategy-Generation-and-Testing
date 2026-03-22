#!/usr/bin/env python3
"""
EXPERIMENT #040 - KAMA StochRSI Pullback with ADX Trend Filter (1h Primary)
==================================================================================================
Hypothesis: Current best uses 4h+1d with HMA+RSI. This uses 1h+4h with KAMA+StochRSI+ADX.

Key innovations:
1. 1h PRIMARY + 4h HTF: More trades than 4h, cleaner signals than 15m/30m
2. KAMA for trend: Adaptive to volatility, performs better in ranging markets than HMA/EMA
3. StochRSI for entries: More sensitive than regular RSI for precise pullback detection
4. ADX trend strength filter: Only trade when ADX > 25 (strong trend), avoid choppy markets
5. Conservative sizing: 0.25 base, 0.35 high conviction, 2.5 ATR stoploss

Why this should beat hma_rsi_pullback_daily_trend_4h_v1 (Sharpe=0.537):
- KAMA adapts to market regime better than static HMA
- StochRSI catches pullbacks earlier than regular RSI
- ADX filter eliminates low-quality trades in ranging markets
- 1h timeframe captures more opportunities while 4h filter keeps quality high
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_stochrsi_adx_pullback_1h_4h_v1"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility - smooth in trends, responsive in ranges
    ER = |Close - Close(n)| / Sum(|Close(i) - Close(i-1)|)
    SC = [ER * (fast - slow) + slow]^2
    KAMA = KAMA(prev) + SC * (Close - KAMA(prev))
    """
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 0:
            er[i] = price_change / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    first_valid = period + slow_period
    kama[first_valid - 1] = close[first_valid - 1]
    
    # Calculate KAMA
    for i in range(first_valid, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
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


def calculate_stoch_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    """
    Stochastic RSI - more sensitive than regular RSI
    StochRSI = (RSI - min(RSI)) / (max(RSI) - min(RSI))
    """
    n = len(close)
    if n < rsi_period + stoch_period:
        return np.zeros(n), np.zeros(n)
    
    rsi = calculate_rsi(close, rsi_period)
    
    stoch_k = np.zeros(n)
    stoch_d = np.zeros(n)
    
    for i in range(rsi_period + stoch_period - 1, n):
        rsi_window = rsi[i - stoch_period + 1:i + 1]
        min_rsi = np.min(rsi_window)
        max_rsi = np.max(rsi_window)
        
        if max_rsi - min_rsi > 0:
            stoch_k[i] = 100 * (rsi[i] - min_rsi) / (max_rsi - min_rsi)
        else:
            stoch_k[i] = 50
    
    # %D is SMA of %K
    for i in range(rsi_period + stoch_period - 1 + d_period - 1, n):
        stoch_d[i] = np.mean(stoch_k[i - d_period + 1:i + 1])
    
    return stoch_k, stoch_d


def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) - measures trend strength
    ADX > 25 = strong trend, ADX < 20 = ranging market
    """
    n = len(close)
    if n < period * 3:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
    # Smooth with Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initialize sums
    sum_tr = np.sum(tr[1:period + 1])
    sum_plus_dm = np.sum(plus_dm[1:period + 1])
    sum_minus_dm = np.sum(minus_dm[1:period + 1])
    
    for i in range(period, n):
        if i == period:
            sum_tr = np.sum(tr[1:i + 1])
            sum_plus_dm = np.sum(plus_dm[1:i + 1])
            sum_minus_dm = np.sum(minus_dm[1:i + 1])
        else:
            sum_tr = sum_tr - sum_tr / period + tr[i]
            sum_plus_dm = sum_plus_dm - sum_plus_dm / period + plus_dm[i]
            sum_minus_dm = sum_minus_dm - sum_minus_dm / period + minus_dm[i]
        
        if sum_tr > 0:
            plus_di[i] = 100 * sum_plus_dm / sum_tr
            minus_di[i] = 100 * sum_minus_dm / sum_tr
        else:
            plus_di[i] = 0
            minus_di[i] = 0
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # ADX is SMA of DX
    for i in range(period * 2, n):
        adx[i] = np.mean(dx[i - period + 1:i + 1])
    
    return adx


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
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    kama_1h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_fast_1h = calculate_kama(close, period=5, fast_period=2, slow_period=15)
    stoch_k_1h, stoch_d_1h = calculate_stoch_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3)
    adx_1h = calculate_adx(high, low, close, period=14)
    supertrend_1h, st_trend_1h = calculate_supertrend(high, low, close, atr_1h, multiplier=3.0)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h KAMA for trend direction
        kama_4h = calculate_kama(close_4h, period=10, fast_period=2, slow_period=30)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
        
        # Align to 1h timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
        st_trend_4h_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.25    # Base position (25% of capital)
    SIZE_HIGH = 0.35    # High conviction (35% of capital)
    
    # ATR stoploss - slightly wider to avoid premature exits
    ATR_STOP_MULT = 2.5
    
    # StochRSI pullback zones
    STOCH_LONG_MIN = 20
    STOCH_LONG_MAX = 50
    STOCH_SHORT_MIN = 50
    STOCH_SHORT_MAX = 80
    
    # ADX trend strength threshold
    ADX_MIN = 25  # Only trade when trend is strong
    
    first_valid = max(150, 100)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(stoch_k_1h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        stoch_k = stoch_k_1h[i]
        stoch_d = stoch_d_1h[i]
        st_trend_val = st_trend_1h[i]
        kama_val = kama_1h[i]
        kama_fast_val = kama_fast_1h[i]
        adx_val = adx_1h[i]
        
        # 4h trend filters (MASTER FILTER)
        kama_4h_val = kama_4h_aligned[i]
        st_trend_4h_val = st_trend_4h_aligned[i]
        adx_4h_val = adx_4h_aligned[i]
        
        # Determine 4h trend direction
        four_h_trend = 0
        if kama_4h_val > 0 and price > kama_4h_val:
            four_h_trend = 1
        elif kama_4h_val > 0 and price < kama_4h_val:
            four_h_trend = -1
        
        if st_trend_4h_val == 1:
            four_h_trend = max(four_h_trend, 1)
        elif st_trend_4h_val == -1:
            four_h_trend = min(four_h_trend, -1)
        
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
        
        # ========== ENTRY LOGIC - STOCHRSI PULLBACK IN TREND DIRECTION ==========
        # LONG: 4h trend up + 1h Supertrend up + StochRSI pullback + ADX strong
        long_condition = (
            four_h_trend == 1 and
            st_trend_val == 1 and
            stoch_k >= STOCH_LONG_MIN and stoch_k <= STOCH_LONG_MAX and
            kama_fast_val > kama_val and
            adx_val >= ADX_MIN
        )
        
        # SHORT: 4h trend down + 1h Supertrend down + StochRSI pullback + ADX strong
        short_condition = (
            four_h_trend == -1 and
            st_trend_val == -1 and
            stoch_k >= STOCH_SHORT_MIN and stoch_k <= STOCH_SHORT_MAX and
            kama_fast_val < kama_val and
            adx_val >= ADX_MIN
        )
        
        # High conviction: 4h ADX also strong + 4h Supertrend confirms
        high_conviction_long = long_condition and st_trend_4h_val == 1 and adx_4h_val >= ADX_MIN
        high_conviction_short = short_condition and st_trend_4h_val == -1 and adx_4h_val >= ADX_MIN
        
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