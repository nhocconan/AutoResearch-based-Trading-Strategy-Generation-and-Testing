#!/usr/bin/env python3
"""
EXPERIMENT #075 - MARKET_STRUCTURE_ADX_VOLUME_ENSEMBLE_1H_4H_V1
==================================================================================================
Hypothesis: Switch to 1h/4h timeframe (different from recent 15m/4h) with ADX regime detection
instead of BBW. Add volume confirmation and market structure (HH/HL, LH/LL) for higher quality
entries. This should reduce false breakouts and improve win rate.

Why this should work:
- 1h timeframe has less noise than 15m, fewer false signals
- ADX > 25 = trending regime, ADX < 25 = mean reversion regime (more stable than BBW)
- Volume confirmation filters out low-liquidity breakouts
- Market structure (swing highs/lows) ensures we trade with structure, not against it
- 3-signal voting (HMA, Supertrend, RSI) with volume/market structure filters
- Discrete position sizing (0.0, ±0.20, ±0.28, ±0.35) minimizes fee churn
- Proper stoploss (2*ATR) and take profit (2R reduce, trail at 1R)

Key differences from #074:
- 1h instead of 15m (less noise, fewer trades but higher quality)
- ADX regime instead of BBW percentile
- Volume confirmation required for entries
- Market structure detection (swing points)
- Different indicator parameters optimized for 1h
"""

import numpy as np
import pandas as pd

name = "market_structure_adx_volume_ensemble_1h_4h_v1"
timeframe = "1h"
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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
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
    
    return adx, plus_di, minus_di


def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    n = len(data)
    if n < period:
        return np.zeros(n)
    
    sma = np.zeros(n)
    for i in range(period - 1, n):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    
    return sma


def detect_market_structure(high, low, close, lookback=20):
    """Detect market structure (HH/HL for uptrend, LH/LL for downtrend)"""
    n = len(close)
    structure = np.zeros(n)  # 1 = bullish (HH/HL), -1 = bearish (LH/LL), 0 = neutral
    
    if n < lookback * 2:
        return structure
    
    swing_highs = np.zeros(n)
    swing_lows = np.zeros(n)
    
    for i in range(lookback, n - lookback):
        if high[i] == np.max(high[i - lookback:i + lookback + 1]):
            swing_highs[i] = 1
        if low[i] == np.min(low[i - lookback:i + lookback + 1]):
            swing_lows[i] = 1
    
    last_sh = 0
    last_sl = 0
    prev_sh = 0
    prev_sl = 0
    
    for i in range(lookback * 2, n):
        if swing_highs[i] == 1:
            if high[i] > last_sh and last_sh > 0:
                structure[i] = 1  # Higher High
            last_sh = high[i]
        if swing_lows[i] == 1:
            if low[i] > last_sl and last_sl > 0:
                structure[i] = 1  # Higher Low
            elif low[i] < last_sl and last_sl > 0:
                structure[i] = -1  # Lower Low
            last_sl = low[i]
    
    # Forward fill structure signal
    for i in range(1, n):
        if structure[i] == 0:
            structure[i] = structure[i - 1]
    
    return structure


def resample_to_timeframe(close, high, low, open_price, volume, bars_per_tf):
    """Resample data to higher timeframe"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    o_tf = np.zeros(n_tf)
    v_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        if end_idx <= n:
            c_tf[i] = close[end_idx - 1]
            h_tf[i] = np.max(high[start_idx:end_idx])
            l_tf[i] = np.min(low[start_idx:end_idx])
            o_tf[i] = open_price[start_idx]
            v_tf[i] = np.sum(volume[start_idx:end_idx])
    
    return c_tf, h_tf, l_tf, o_tf, v_tf


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_price = prices["open"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    if n < 500:
        return np.zeros(n)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_fast_1h = calculate_hma(close, period=16)
    hma_slow_1h = calculate_hma(close, period=48)
    st_1h, st_upper_1h, st_lower_1h, st_trend_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(high, low, close, period=14)
    volume_sma_1h = calculate_sma(volume, 20)
    market_struct_1h = detect_market_structure(high, low, close, lookback=20)
    
    # Resample to 4h for trend regime (4 x 1h = 4h)
    bars_per_4h = 4
    c_4h, h_4h, l_4h, o_4h, v_4h = resample_to_timeframe(close, high, low, open_price, volume, bars_per_4h)
    n_4h = len(c_4h)
    
    if n_4h < 50:
        return np.zeros(n)
    
    # 4h indicators for trend regime
    hma_fast_4h = calculate_hma(c_4h, period=16)
    hma_slow_4h = calculate_hma(c_4h, period=48)
    st_4h, st_upper_4h, st_lower_4h, st_trend_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    
    # Calculate ADX regime threshold
    adx_valid = adx_4h[50:]
    adx_threshold = np.median(adx_valid[adx_valid > 0]) if len(adx_valid[adx_valid > 0]) > 0 else 25.0
    
    # Map 4h indicators back to 1h timeframe
    hma_trend_4h = np.zeros(n)
    st_trend_4h_mapped = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    regime = np.zeros(n)  # 0 = low ADX (mean revert), 1 = high ADX (trend)
    
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
            
            # ADX value and regime
            adx_4h_mapped[i] = adx_4h[idx_4h]
            regime[i] = 1 if adx_4h[idx_4h] > adx_threshold else 0
            
            # ATR value
            atr_4h_mapped[i] = atr_4h[idx_4h]
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_LOW = 0.20
    SIZE_MED = 0.28
    SIZE_HIGH = 0.35
    
    # Thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    RSI_MR_LONG = 30
    RSI_MR_SHORT = 70
    ATR_STOP_MULT = 2.5
    ADX_TREND_THRESHOLD = 25
    
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
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            if pos_side != 0:
                pos_side = 0
                pos_entry = 0.0
                pos_tp_triggered = False
                pos_highest = 0.0
                pos_lowest = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        rsi = rsi_1h[i]
        vol = volume[i]
        vol_sma = volume_sma_1h[i]
        mstruct = market_struct_1h[i]
        
        # Get regime info
        hma_trend = hma_trend_4h[i]
        st_trend = st_trend_4h_mapped[i]
        adx_val = adx_4h_mapped[i]
        regime_val = regime[i]
        atr_4h_val = atr_4h_mapped[i]
        
        # Volume confirmation
        volume_confirmed = vol > vol_sma * 1.0 if vol_sma > 0 else False
        
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
                    signals[i] = SIZE_LOW * 0.5
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
                    signals[i] = -SIZE_LOW * 0.5
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
        # Count signal agreement (3-signal voting)
        confidence = 0
        signal_direction = 0
        
        # Signal 1: HMA trend (4h)
        if hma_trend != 0:
            confidence += 1
            signal_direction += hma_trend
        
        # Signal 2: Supertrend (4h)
        if st_trend != 0:
            confidence += 1
            signal_direction += int(st_trend)
        
        # Signal 3: 1h RSI momentum
        if rsi > 50:
            confidence += 1
            signal_direction += 1
        elif rsi < 50:
            confidence += 1
            signal_direction -= 1
        
        # Determine position size based on confidence
        if confidence >= 3:
            base_size = SIZE_HIGH
        elif confidence >= 2:
            base_size = SIZE_MED
        else:
            signals[i] = 0.0
            continue
        
        # Entry logic based on regime
        if regime_val == 1:
            # HIGH ADX (trending regime) - TREND FOLLOWING
            # LONG: uptrend + RSI pullback + volume + market structure
            if signal_direction >= 2 and hma_trend == 1 and st_trend == 1:
                if RSI_LONG_MIN <= rsi <= RSI_LONG_MAX:
                    if volume_confirmed and (mstruct >= 0):
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
            
            # SHORT: downtrend + RSI pullback + volume + market structure
            elif signal_direction <= -2 and hma_trend == -1 and st_trend == -1:
                if RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX:
                    if volume_confirmed and (mstruct <= 0):
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
        
        else:
            # LOW ADX (mean reversion regime) - MEAN REVERSION
            # LONG: RSI oversold in uptrend + volume
            if hma_trend == 1 and rsi <= RSI_MR_LONG:
                if volume_confirmed:
                    signals[i] = SIZE_MED
                    pos_side = 1
                    pos_entry = price
                    pos_entry_bar = i
                    pos_tp_triggered = False
                    pos_highest = price
                    pos_lowest = price
                else:
                    signals[i] = 0.0
            # SHORT: RSI overbought in downtrend + volume
            elif hma_trend == -1 and rsi >= RSI_MR_SHORT:
                if volume_confirmed:
                    signals[i] = -SIZE_MED
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
    
    return signals