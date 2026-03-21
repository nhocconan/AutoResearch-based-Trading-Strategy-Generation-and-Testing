#!/usr/bin/env python3
"""
EXPERIMENT #004 - KAMA Adaptive Trend + RSI Pullback with 4h Filter (1h Primary)
==================================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than fixed SMAs,
reducing whipsaws in choppy markets. Combined with RSI pullback entries and 4h trend filter.

Key improvements over #003 (which crashed):
1. SIMPLER position tracking - no complex state arrays that caused indexing errors
2. Primary=1h, HTF=4h (same combo as #002 which achieved Sharpe=0.128)
3. KAMA instead of Supertrend - smoother trend following, fewer false flips
4. RSI pullback (not MACD) - proven to work better for mean-reversion entries in trends
5. ADX filter - only trade when trend strength > 20 (avoids chop)

Why KAMA > Supertrend for this strategy:
- KAMA adjusts smoothing based on market noise ratio
- Supertrend flips too frequently in ranging markets
- KAMA + RSI pullback = enter on dips in established trends

Risk Management:
- Position size: 0.20-0.30 (discrete, conservative)
- Stoploss: 2.5 ATR trailing stop via signal→0
- Take profit: Reduce to half at 2R, trail at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_adx_mtf_1h_4h_v1"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market noise/volatility
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = price_change / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    for i in range(er_period, n):
        sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
        sc[i] = sc[i] ** 2  # Square for smoother adaptation
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
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
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    atr[period - 1] = np.mean(tr[1:period])
    plus_di[period - 1] = 100 * np.mean(plus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    minus_di[period - 1] = 100 * np.mean(minus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_di[i] = 100 * ((plus_di[i - 1] * (period - 1) + 100 * plus_dm[i] / atr[i] if atr[i] > 0 else 0)) / period
        minus_di[i] = 100 * ((minus_di[i - 1] * (period - 1) + 100 * minus_dm[i] / atr[i] if atr[i] > 0 else 0)) / period
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # ADX is SMA of DX
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx


def calculate_kama_trend(kama, close):
    """Determine trend direction from KAMA slope"""
    n = len(close)
    trend = np.zeros(n)
    
    for i in range(1, n):
        if kama[i] > kama[i - 1]:
            trend[i] = 1
        elif kama[i] < kama[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    
    return trend


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h INDICATORS (ENTRY TIMING) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_trend_1h = calculate_kama_trend(kama_1h, close)
    rsi_1h = calculate_rsi(close, period=14)
    adx_1h = calculate_adx(high, low, close, period=14)
    
    # Fast KAMA for crossover signals
    kama_fast_1h = calculate_kama(close, er_period=5, fast_period=2, slow_period=20)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h KAMA for master trend
        kama_4h = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
        kama_trend_4h = calculate_kama_trend(kama_4h, close_4h)
        
        # 4h ADX for trend strength filter
        adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
        
        # Align to 1h timeframe (auto shift for completed bars)
        kama_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_trend_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        
    except Exception:
        kama_trend_4h_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.20
    SIZE_HIGH = 0.30
    
    # Filter thresholds
    ADX_MIN = 20.0  # Minimum trend strength
    RSI_LONG_ENTRY = 45.0  # RSI pullback level for longs
    RSI_SHORT_ENTRY = 55.0  # RSI pullback level for shorts
    
    # ATR stoploss
    ATR_STOP_MULT = 2.5
    TP_MULT = 2.0
    TRAIL_MULT = 1.0
    
    first_valid = max(200, 100)
    
    # Track position state (simplified)
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_trend_1h[i]) or np.isnan(kama_trend_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        kama_trend = kama_trend_1h[i]
        rsi = rsi_1h[i]
        adx = adx_1h[i]
        
        # 4h trend filter (master)
        trend_4h = kama_trend_4h_aligned[i]
        adx_4h = adx_4h_aligned[i]
        
        # ========== CHECK EXISTING POSITIONS ==========
        if in_position:
            # Update highest/lowest since entry
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, price)
                lowest_since_entry = min(lowest_since_entry, price) if lowest_since_entry > 0 else price
            else:
                highest_since_entry = max(highest_since_entry, price) if highest_since_entry > 0 else price
                lowest_since_entry = min(lowest_since_entry, price)
            
            # KAMA trend flip stoploss
            if position_side == 1 and kama_trend == -1:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
            
            if position_side == -1 and kama_trend == 1:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
            
            # ATR stoploss check
            if position_side == 1:
                stoploss_price = entry_price - ATR_STOP_MULT * entry_atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check - reduce to half
                tp_price = entry_price + TP_MULT * entry_atr
                if not tp_triggered and price >= tp_price:
                    signals[i] = SIZE_BASE / 2
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R profit
                if tp_triggered:
                    trail_stop = highest_since_entry - TRAIL_MULT * entry_atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
                    
            elif position_side == -1:
                stoploss_price = entry_price + ATR_STOP_MULT * entry_atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check - reduce to half
                tp_price = entry_price - TP_MULT * entry_atr
                if not tp_triggered and price <= tp_price:
                    signals[i] = -SIZE_BASE / 2
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R profit
                if tp_triggered:
                    trail_stop = lowest_since_entry + TRAIL_MULT * entry_atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
            
            # Hold position
            signals[i] = signals[i - 1]
            continue
        
        # ========== NEW ENTRY LOGIC ==========
        # 4h trend must be established (ADX > 20)
        if adx_4h < ADX_MIN:
            signals[i] = 0.0
            continue
        
        # 4h and 1h trends must align
        if trend_4h == 0 or kama_trend == 0:
            signals[i] = 0.0
            continue
        
        if trend_4h != kama_trend:
            signals[i] = 0.0
            continue
        
        # RSI pullback entry
        long_setup = (
            trend_4h == 1 and
            kama_trend == 1 and
            rsi < RSI_LONG_ENTRY and
            rsi > 30  # Not oversold
        )
        
        short_setup = (
            trend_4h == -1 and
            kama_trend == -1 and
            rsi > RSI_SHORT_ENTRY and
            rsi < 70  # Not overbought
        )
        
        # Determine position size based on ADX strength
        high_conviction = adx_4h > 30 and adx > 25
        
        if long_setup:
            size = SIZE_HIGH if high_conviction else SIZE_BASE
            signals[i] = size
            in_position = True
            position_side = 1
            entry_price = price
            entry_atr = atr
            tp_triggered = False
            highest_since_entry = price
            lowest_since_entry = price
        
        elif short_setup:
            size = SIZE_HIGH if high_conviction else SIZE_BASE
            signals[i] = -size
            in_position = True
            position_side = -1
            entry_price = price
            entry_atr = atr
            tp_triggered = False
            highest_since_entry = price
            lowest_since_entry = price
        
        else:
            signals[i] = 0.0
    
    return signals