#!/usr/bin/env python3
"""
EXPERIMENT #073 - VOLATILITY_ADAPTIVE_HMA_ST_MOMENTUM_15M_4H_V1
==================================================================================================
Hypothesis: After #072's moderate Sharpe (0.589), return to 15m timeframe (proven in #062 with 
Sharpe=11.924) but with improved regime detection and position sizing. Use ATR volatility 
percentile instead of BBW for regime, add ROC momentum confirmation, and scale position size 
by both signal confidence AND current volatility level.

Why this should work:
- 15m has more trade opportunities than 1h (proven in #062)
- ATR volatility regime is more direct than BBW for position sizing
- ROC(10) momentum filter reduces false breakouts in choppy markets
- Volatility-adaptive sizing: smaller positions in high vol, larger in low vol
- HMA(16/48) + Supertrend(10,3) proven combination from best strategies
- 4h trend filter prevents counter-trend trades (learned from #061)
- Discrete signal levels (0, ±0.20, ±0.30, ±0.35) minimize fee churn
- Clean position state tracking (no scoping issues from #069)

Key changes from #072:
- 15m instead of 1h (more signals, proven track record)
- ATR volatility regime instead of BBW percentile
- ROC momentum confirmation filter
- Position size scales with volatility (smaller in high vol)
- Added Donchian channel for breakout confirmation
"""

import numpy as np
import pandas as pd

name = "volatility_adaptive_hma_st_momentum_15m_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, wma_period):
        n_wma = len(data)
        result = np.zeros(n_wma)
        weights = np.arange(1, wma_period + 1)
        weight_sum = np.sum(weights)
        
        for i in range(wma_period - 1, n_wma):
            window = data[i - wma_period + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        
        return result
    
    wma_full = wma(close, period)
    wma_half = wma(close, half_period)
    
    hma_raw = 2 * wma_half - wma_full
    
    hma = np.zeros(n)
    weights = np.arange(1, sqrt_period + 1)
    weight_sum = np.sum(weights)
    
    for i in range(sqrt_period - 1, n):
        if i >= len(hma_raw):
            break
        start_idx = max(0, i - sqrt_period + 1)
        window = hma_raw[start_idx:i + 1]
        if len(window) == sqrt_period:
            hma[i] = np.sum(window * weights) / weight_sum
    
    return hma


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]
    
    supertrend[period] = upper_band[period]
    trend[period] = -1 if close[period] < supertrend[period] else 1
    
    for i in range(period + 1, n):
        if trend[i - 1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, upper_band, lower_band, trend


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


def calculate_roc(close, period=10):
    """Calculate Rate of Change (momentum)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    roc = np.zeros(n)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100
    
    return roc


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


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
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    for i in range(period, n):
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


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion detection"""
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
    
    return zscore


def resample_to_timeframe(close, high, low, open_price, bars_per_tf):
    """Resample data to higher timeframe"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    o_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        if end_idx <= n:
            c_tf[i] = close[end_idx - 1]
            h_tf[i] = np.max(high[start_idx:end_idx])
            l_tf[i] = np.min(low[start_idx:end_idx])
            o_tf[i] = open_price[start_idx]
    
    return c_tf, h_tf, l_tf, o_tf


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_price = prices["open"].values
    n = len(close)
    
    if n < 500:
        return np.zeros(n)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_fast_15m = calculate_hma(close, period=16)
    hma_slow_15m = calculate_hma(close, period=48)
    st_15m, st_upper_15m, st_lower_15m, st_trend_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    roc_15m = calculate_roc(close, period=10)
    donchian_upper_15m, donchian_lower_15m = calculate_donchian(high, low, period=20)
    adx_15m = calculate_adx(high, low, close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    
    # Resample to 4h for trend regime (16 x 15m = 4h)
    bars_per_4h = 16
    c_4h, h_4h, l_4h, o_4h = resample_to_timeframe(close, high, low, open_price, bars_per_4h)
    n_4h = len(c_4h)
    
    if n_4h < 50:
        return np.zeros(n)
    
    # 4h indicators for trend regime
    hma_fast_4h = calculate_hma(c_4h, period=16)
    hma_slow_4h = calculate_hma(c_4h, period=48)
    st_4h, st_upper_4h, st_lower_4h, st_trend_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Calculate ATR volatility percentile for regime detection
    atr_valid = atr_4h[50:]
    if len(atr_valid) > 0:
        atr_sorted = np.sort(atr_valid)
    else:
        atr_sorted = np.array([0])
    
    # Map 4h indicators back to 15m timeframe
    hma_trend_4h = np.zeros(n)
    st_trend_4h_mapped = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    vol_regime = np.zeros(n)  # 0 = low vol, 1 = high vol
    atr_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 50:
            # HMA trend
            if c_4h[idx_4h] > hma_slow_4h[idx_4h] and hma_fast_4h[idx_4h] > hma_slow_4h[idx_4h]:
                hma_trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_slow_4h[idx_4h] and hma_fast_4h[idx_4h] < hma_slow_4h[idx_4h]:
                hma_trend_4h[i] = -1
            
            # Supertrend
            st_trend_4h_mapped[i] = st_trend_4h[idx_4h]
            
            # ADX strength
            adx_4h_mapped[i] = adx_4h[idx_4h]
            
            # ATR value
            atr_4h_mapped[i] = atr_4h[idx_4h]
            
            # Volatility regime (percentile-based)
            atr_idx = np.searchsorted(atr_sorted, atr_4h[idx_4h]) / len(atr_sorted)
            vol_regime[i] = 1 if atr_idx > 0.7 else 0
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels with volatility adjustment
    SIZE_LOW = 0.20
    SIZE_MED = 0.28
    SIZE_HIGH = 0.35
    
    # Thresholds
    ADX_MIN = 20
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    ATR_STOP_MULT = 2.5
    ROC_MIN = 0.5  # Minimum momentum for entry
    ZSCORE_MAX = 1.5  # Max z-score for entry (avoid extremes)
    
    first_valid = max(300, 50 * bars_per_4h)
    
    # Position state tracking
    pos_side = 0
    pos_entry = 0.0
    pos_entry_bar = 0
    pos_tp_triggered = False
    pos_highest = 0.0
    pos_lowest = 0.0
    
    for i in range(first_valid, n):
        # Check for invalid data
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0 or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            pos_side = 0
            pos_entry = 0.0
            pos_tp_triggered = False
            pos_highest = 0.0
            pos_lowest = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi = rsi_15m[i]
        roc = roc_15m[i]
        zscore = zscore_15m[i]
        
        # Get regime info
        hma_trend = hma_trend_4h[i]
        st_trend = st_trend_4h_mapped[i]
        adx_val = adx_4h_mapped[i]
        vol_regime_val = vol_regime[i]
        atr_4h_val = atr_4h_mapped[i]
        
        # Manage existing position
        if pos_side != 0:
            # Update highest/lowest since entry
            if pos_side == 1:
                pos_highest = max(pos_highest, price) if pos_highest > 0 else price
            else:
                pos_lowest = min(pos_lowest, price) if pos_lowest > 0 else price
            
            # Stoploss check
            if pos_side == 1:
                stoploss_price = pos_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    pos_side = 0
                    pos_entry = 0.0
                    pos_tp_triggered = False
                    pos_highest = 0.0
                    pos_lowest = 0.0
                    continue
                
                # Take profit (2R) - reduce to half
                tp_price = pos_entry + 2 * ATR_STOP_MULT * atr
                if not pos_tp_triggered and price >= tp_price:
                    signals[i] = SIZE_LOW * 0.5 if vol_regime_val == 0 else SIZE_LOW * 0.35
                    pos_tp_triggered = True
                    continue
                
                # Trail stop at 1R
                if pos_tp_triggered:
                    trail_stop = pos_highest - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        pos_side = 0
                        pos_entry = 0.0
                        pos_tp_triggered = False
                        pos_highest = 0.0
                        pos_lowest = 0.0
                        continue
            
            elif pos_side == -1:
                stoploss_price = pos_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    pos_side = 0
                    pos_entry = 0.0
                    pos_tp_triggered = False
                    pos_highest = 0.0
                    pos_lowest = 0.0
                    continue
                
                # Take profit (2R) - reduce to half
                tp_price = pos_entry - 2 * ATR_STOP_MULT * atr
                if not pos_tp_triggered and price <= tp_price:
                    signals[i] = -SIZE_LOW * 0.5 if vol_regime_val == 0 else -SIZE_LOW * 0.35
                    pos_tp_triggered = True
                    continue
                
                # Trail stop at 1R
                if pos_tp_triggered:
                    trail_stop = pos_lowest + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        pos_side = 0
                        pos_entry = 0.0
                        pos_tp_triggered = False
                        pos_highest = 0.0
                        pos_lowest = 0.0
                        continue
            
            # Hold position
            signals[i] = signals[i - 1] if i > 0 else 0.0
            continue
        
        # No position - check for new entry
        # Count signal agreement
        confidence = 0
        signal_direction = 0
        
        # Signal 1: HMA trend (4h)
        if hma_trend != 0:
            confidence += 1
            signal_direction += hma_trend
        
        # Signal 2: Supertrend (4h)
        if st_trend != 0:
            confidence += 1
            signal_direction += st_trend
        
        # Signal 3: ADX strength
        if adx_val >= ADX_MIN:
            confidence += 1
        
        # Signal 4: ROC momentum confirmation
        if abs(roc) >= ROC_MIN:
            confidence += 1
            signal_direction += 1 if roc > 0 else -1
        
        # Signal 5: Z-score filter (not at extreme)
        if abs(zscore) <= ZSCORE_MAX:
            confidence += 1
        
        # Determine position size based on confidence AND volatility
        if confidence >= 5:
            base_size = SIZE_HIGH
        elif confidence >= 4:
            base_size = SIZE_MED
        elif confidence >= 3:
            base_size = SIZE_LOW
        else:
            signals[i] = 0.0
            continue
        
        # Volatility adjustment: reduce size in high volatility
        if vol_regime_val == 1:
            base_size *= 0.7  # 30% reduction in high vol
        
        # Entry logic
        # LONG: uptrend + RSI pullback + momentum confirmation
        if signal_direction >= 2 and hma_trend == 1 and st_trend == 1:
            if RSI_LONG_MIN <= rsi <= RSI_LONG_MAX and roc >= ROC_MIN:
                # Additional filter: price above Donchian mid
                donchian_mid = (donchian_upper_15m[i] + donchian_lower_15m[i]) / 2
                if price > donchian_mid:
                    signals[i] = base_size
                    pos_side = 1
                    pos_entry = price
                    pos_entry_bar = i
                    pos_tp_triggered = False
                    pos_highest = price
                    pos_lowest = price
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        # SHORT: downtrend + RSI pullback + momentum confirmation
        elif signal_direction <= -2 and hma_trend == -1 and st_trend == -1:
            if RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX and roc <= -ROC_MIN:
                # Additional filter: price below Donchian mid
                donchian_mid = (donchian_upper_15m[i] + donchian_lower_15m[i]) / 2
                if price < donchian_mid:
                    signals[i] = -base_size
                    pos_side = -1
                    pos_entry = price
                    pos_entry_bar = i
                    pos_tp_triggered = False
                    pos_highest = price
                    pos_lowest = price
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals