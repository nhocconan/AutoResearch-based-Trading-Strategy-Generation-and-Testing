#!/usr/bin/env python3
"""
EXPERIMENT #014 - EMA Momentum with ADX Regime Filter (30m Primary)
==================================================================================================
Hypothesis: 30m primary timeframe provides more entry opportunities than 1h while maintaining
strict 4h+1d trend alignment. EMA crossover gives faster signals than KAMA/HMA. ADX filter
avoids choppy regimes that caused whipsaws in previous strategies.

Key innovations:
1. Primary=30m (faster than 1h, more signals than 4h)
2. EMA(8/21) crossover for entry timing (faster than KAMA)
3. ADX(14) > 25 filter to avoid choppy markets (major improvement over BBW filter)
4. 4h EMA trend + 1d EMA master trend alignment
5. RSI momentum confirmation (40-60 zone for entries, avoids extremes)
6. ATR trailing stop with take-profit scaling

Why this should beat hma_rsi_pullback_daily_trend_4h_v1 (Sharpe=0.537):
- 30m provides 2x more entry opportunities than 1h
- ADX filter is more reliable than BBW for trend regime detection
- EMA crossover is faster than HMA for entry timing
- Triple timeframe alignment (30m+4h+1d) reduces false signals
- Conservative position sizing (0.25 base, 0.35 high conviction)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ema_momentum_adx_regime_30m_4h_1d_v1"
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


def calculate_ema(close, period=14):
    """Calculate EMA with proper min_periods"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    return ema


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
    Calculate ADX (Average Directional Index)
    Measures trend strength regardless of direction
    ADX > 25 = strong trend, ADX < 20 = choppy/ranging
    """
    n = len(close)
    if n < period * 3:
        return np.zeros(n)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's method
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    atr[period - 1] = np.mean(tr[1:period])
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    plus_dm_smooth[period - 1] = np.mean(plus_dm[1:period])
    minus_dm_smooth[period - 1] = np.mean(minus_dm[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_dm_smooth[i] = (plus_dm_smooth[i - 1] * (period - 1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i - 1] * (period - 1) + minus_dm[i]) / period
        
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # ========== 30m INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    ema_8_30m = calculate_ema(close, period=8)
    ema_21_30m = calculate_ema(close, period=21)
    ema_50_30m = calculate_ema(close, period=50)
    adx_30m = calculate_adx(high, low, close, period=14)
    vol_ma_30m = calculate_volume_ma(volume, period=20)
    
    # ========== 4h INDICATORS (INTERMEDIATE TREND) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        ema_21_4h = calculate_ema(close_4h, period=21)
        ema_50_4h = calculate_ema(close_4h, period=50)
        adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
        
        # Align to 30m timeframe (auto shift for completed bars)
        ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
        ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        
    except Exception:
        ema_21_4h_aligned = np.zeros(n)
        ema_50_4h_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
    
    # ========== 1d INDICATORS (MASTER TREND FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        ema_21_1d = calculate_ema(close_1d, period=21)
        ema_50_1d = calculate_ema(close_1d, period=50)
        adx_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
        
        # Align to 30m timeframe (auto shift for completed bars)
        ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
        
    except Exception:
        ema_21_1d_aligned = np.zeros(n)
        ema_50_1d_aligned = np.zeros(n)
        adx_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE & DISCRETE
    SIZE_BASE = 0.25   # Base position (25%)
    SIZE_HIGH = 0.35   # High conviction (35%)
    SIZE_MAX = 0.40    # Maximum position (absolute max)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.5
    
    # ADX threshold (avoid choppy markets)
    ADX_MIN = 22
    
    # RSI momentum zones
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # EMA crossover confirmation
    EMA_CROSS_MIN_GAP = 0.001  # 0.1% minimum gap between EMAs
    
    first_valid = max(100, 50)
    
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
        adx_val = adx_30m[i]
        ema_8_val = ema_8_30m[i]
        ema_21_val = ema_21_30m[i]
        ema_50_val = ema_50_30m[i]
        vol_ratio = volume[i] / vol_ma_30m[i] if vol_ma_30m[i] > 0 else 1.0
        
        # 4h trend filters
        ema_21_4h_val = ema_21_4h_aligned[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        adx_4h_val = adx_4h_aligned[i]
        
        # 1d trend filters (MASTER FILTER)
        ema_21_1d_val = ema_21_1d_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        adx_1d_val = adx_1d_aligned[i]
        
        # Skip if HTF data not available
        if ema_21_4h_val == 0 or ema_21_1d_val == 0:
            signals[i] = 0.0
            continue
        
        # Determine trend directions
        # 1d master trend
        daily_trend = 0
        if ema_21_1d_val > 0 and ema_50_1d_val > 0:
            if ema_21_1d_val > ema_50_1d_val and price > ema_21_1d_val:
                daily_trend = 1
            elif ema_21_1d_val < ema_50_1d_val and price < ema_21_1d_val:
                daily_trend = -1
        
        # 4h intermediate trend
        h4_trend = 0
        if ema_21_4h_val > 0 and ema_50_4h_val > 0:
            if ema_21_4h_val > ema_50_4h_val and price > ema_21_4h_val:
                h4_trend = 1
            elif ema_21_4h_val < ema_50_4h_val and price < ema_21_4h_val:
                h4_trend = -1
        
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
        
        # ========== REGIME FILTER (ADX) ==========
        # Only trade when ADX indicates strong trend (avoid choppy markets)
        # Check both 30m and 4h ADX for confirmation
        regime_ok = adx_val >= ADX_MIN or adx_4h_val >= ADX_MIN
        
        # Volume confirmation (above 0.8x average)
        volume_ok = vol_ratio >= 0.8
        
        if not regime_ok or not volume_ok:
            signals[i] = 0.0
            continue
        
        # ========== ENTRY LOGIC - TRIPLE TIMEFRAME ALIGNMENT ==========
        # LONG: 1d trend up + 4h trend up + 30m EMA crossover + RSI momentum + ADX confirmation
        ema_cross_long = ema_8_val > ema_21_val and (ema_8_val - ema_21_val) / price > EMA_CROSS_MIN_GAP
        ema_trend_long = ema_21_val > ema_50_val and price > ema_21_val
        
        long_condition = (
            daily_trend == 1 and
            h4_trend == 1 and
            ema_trend_long and
            ema_cross_long and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            adx_val >= ADX_MIN
        )
        
        # SHORT: 1d trend down + 4h trend down + 30m EMA crossover + RSI momentum + ADX confirmation
        ema_cross_short = ema_8_val < ema_21_val and (ema_21_val - ema_8_val) / price > EMA_CROSS_MIN_GAP
        ema_trend_short = ema_21_val < ema_50_val and price < ema_21_val
        
        short_condition = (
            daily_trend == -1 and
            h4_trend == -1 and
            ema_trend_short and
            ema_cross_short and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            adx_val >= ADX_MIN
        )
        
        # Determine position size based on conviction
        # High conviction: all three timeframes align + strong volume + strong ADX
        high_conviction_long = (
            long_condition and 
            adx_4h_val >= ADX_MIN and 
            adx_1d_val >= ADX_MIN and 
            vol_ratio >= 1.2
        )
        high_conviction_short = (
            short_condition and 
            adx_4h_val >= ADX_MIN and 
            adx_1d_val >= ADX_MIN and 
            vol_ratio >= 1.2
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