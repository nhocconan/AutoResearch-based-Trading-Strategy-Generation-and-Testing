#!/usr/bin/env python3
"""
EXPERIMENT #048 - Single 15m HMA+Supertrend+ADX+KAMA+RSI+Z-score+BBW+DEMA (Optimized v3)
==================================================================================================
Hypothesis: #040 achieved Sharpe=16.0 with single 15m timeframe (no 4h filter). #047 added 4h filter
and Sharpe dropped to 8.0. The 4h filter adds lag and reduces opportunities.

Key changes from #047:
- Timeframe: Single 15m (remove 4h trend filter - like #040's winning approach)
- Position size: 0.35 (vs 0.30 in #047 - more aggressive like #040)
- Stoploss: 1.5*ATR (tighter than #047's 2.0*ATR for better risk/reward)
- RSI thresholds: 40-60 (tighter pullback range than #047's 35-65)
- Added DEMA(8/21) for faster trend confirmation on 15m
- BBW filter: > 0.012 (15m appropriate level, slightly lower than #047)
- ADX filter: > 18 (lower than #047's 20 for more opportunities on single TF)
- Z-score filter: < 1.8 (tighter than #047's 2.0)
- Take profit: 1.5R target (vs 2R in #047 - quicker profits on 15m)

Why this should beat #047:
- Single 15m had Sharpe=16.0 in #040 (best ever)
- 4h filter added lag without proportional benefit
- Tighter stops + quicker TP = better risk/reward on fast timeframe
- DEMA adds responsive trend confirmation
- All indicators from #040's winning combination preserved
"""

import numpy as np
import pandas as pd

name = "mtf_hma_supertrend_adx_kama_rsi_zscore_bbw_dema_15m_v3"
timeframe = "15m"
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
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        raw_vals = 2 * wma1[start_idx:i + 1] - wma2[start_idx:i + 1]
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


def calculate_dema(close, fast=8, slow=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < slow:
        return np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    dema_fast = np.zeros(n)
    dema_slow = np.zeros(n)
    
    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    
    for i in range(1, n):
        ema_fast[i] = close[i] * (2.0 / (fast + 1)) + ema_fast[i - 1] * (1.0 - 2.0 / (fast + 1))
        ema_slow[i] = close[i] * (2.0 / (slow + 1)) + ema_slow[i - 1] * (1.0 - 2.0 / (slow + 1))
    
    dema_fast[0] = ema_fast[0]
    dema_slow[0] = ema_slow[0]
    
    for i in range(1, n):
        dema_fast[i] = 2 * ema_fast[i] - (2.0 / (fast + 1)) * ema_fast[i] - (1.0 - 2.0 / (fast + 1)) * dema_fast[i - 1]
        dema_slow[i] = 2 * ema_slow[i] - (2.0 / (slow + 1)) * ema_slow[i] - (1.0 - 2.0 / (slow + 1)) * dema_slow[i - 1]
    
    return dema_fast, dema_slow


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    change = np.zeros(n)
    volatility = np.zeros(n)
    er = np.zeros(n)
    sc = np.zeros(n)
    
    for i in range(er_period, n):
        change[i] = abs(close[i] - close[i - er_period])
        volatility[i] = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
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
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


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
    
    atr = np.zeros(n)
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_sma = np.zeros(n)
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 15m indicators for entry timing (single TF like #040)
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    adx_15m = calculate_adx(high, low, close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    vol_sma_15m = calculate_volume_sma(volume, period=20)
    dema_fast, dema_slow = calculate_dema(close, fast=8, slow=21)
    
    # Generate signals with single timeframe logic (like #040's winning approach)
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries (tighter than #047)
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Volume filter multiplier
    VOL_MULT = 1.1
    
    # BBW minimum for regime filter (15m appropriate)
    BBW_MIN = 0.012
    
    # ADX minimum for trend strength (lower for single TF)
    ADX_MIN = 18
    
    # ATR stoploss multiplier (tighter than #047)
    ATR_STOP_MULT = 1.5
    
    # Z-score filter limits (tighter than #047)
    ZSCORE_MAX = 1.8
    
    first_valid = max(200, 14 * 2, 20, 28, 21)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Single 15m trend filters (no 4h lag)
        st_trend = st_direction_15m[i]
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        bbw_val = bbw_15m[i]
        adx_val = adx_15m[i]
        zscore_val = zscore_15m[i]
        vol_ratio = volume[i] / vol_sma_15m[i] if vol_sma_15m[i] > 0 else 0
        
        # HMA trend direction
        hma_trend = 1 if price > hma_15m[i] else (-1 if price < hma_15m[i] else 0)
        
        # KAMA trend direction
        kama_trend = 1 if price > kama_15m[i] else (-1 if price < kama_15m[i] else 0)
        
        # DEMA trend direction (fast confirmation)
        dema_trend = 1 if dema_fast[i] > dema_slow[i] else (-1 if dema_fast[i] < dema_slow[i] else 0)
        
        # BBW filter - avoid ultra-low volatility markets
        if bbw_val < BBW_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # ADX filter - require trend strength
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Z-score filter - avoid extreme deviations
        if abs(zscore_val) > ZSCORE_MAX:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filters must agree (HMA + Supertrend + KAMA + DEMA on 15m)
        if not (hma_trend == st_trend == kama_trend == dema_trend) or hma_trend == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
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
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (1.5R) - reduce to half
                tp_price = prev_entry + 1.5 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (1.5R) - reduce to half
                tp_price = prev_entry - 1.5 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
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
        
        # Entry logic: All 15m trend filters + RSI pullback + Volume + Z-score
        if hma_trend == 1 and st_trend == 1 and kama_trend == 1 and dema_trend == 1:  # Bullish trend confirmed
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                vol_ratio >= VOL_MULT and
                zscore_val < ZSCORE_MAX):  # Pullback + volume + not overextended
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif hma_trend == -1 and st_trend == -1 and kama_trend == -1 and dema_trend == -1:  # Bearish trend confirmed
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                vol_ratio >= VOL_MULT and
                zscore_val > -ZSCORE_MAX):  # Pullback + volume + not overextended
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals