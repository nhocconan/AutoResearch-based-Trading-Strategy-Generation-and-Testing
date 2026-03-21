#!/usr/bin/env python3
"""
EXPERIMENT #052 - ENSEMBLE VOTING WITH ADX REGIME + VOLUME CONFIRMATION
==================================================================================================
Hypothesis: Adding volume confirmation to ensemble voting will reduce false signals and improve
Sharpe ratio. Using ADX for regime detection (trend strength) instead of BBW percentile.

Key innovations:
- 3 signal types: HMA trend, RSI momentum, Volume confirmation (vote-based)
- Regime detection: ADX(14) > 25 = trend mode, ADX < 20 = mean-revert mode
- Volume filter: only take signals when volume > 20-period average * 0.8
- MTF: 15m entries + 1h trend filter + 4h regime detection
- Conservative sizing: max 0.30, discrete levels (0.0, ±0.15, ±0.25, ±0.30)
- Stoploss: 2*ATR with trailing at 1R profit

Why this should work:
- Volume confirmation filters out low-liquidity false breakouts
- ADX regime detection is more robust than BBW percentile
- Simpler code structure avoids variable scope issues from #051
- Based on lessons from #047 (DD too deep) and #051 (crash)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ensemble_adx_volume_regime_15m_1h_4h_v1"
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
        window = close[i - half_period + 1:i + 1]
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(window * weights) / np.sum(weights)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(window * weights) / np.sum(weights)
    
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        raw_vals = 2 * wma1[start_idx:i + 1] - wma2[start_idx:i + 1]
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


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
    if n < period * 3:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > 0 and high_diff > low_diff:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
        
        if low_diff > 0 and low_diff > high_diff:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    sum_plus_dm = np.sum(plus_dm[1:period + 1])
    sum_minus_dm = np.sum(minus_dm[1:period + 1])
    sum_tr = np.sum(tr[1:period + 1])
    
    for i in range(period, n):
        if i == period:
            smoothed_plus_dm = sum_plus_dm
            smoothed_minus_dm = sum_minus_dm
            smoothed_tr = sum_tr
        else:
            smoothed_plus_dm = smoothed_plus_dm - smoothed_plus_dm / period + plus_dm[i]
            smoothed_minus_dm = smoothed_minus_dm - smoothed_minus_dm / period + minus_dm[i]
            smoothed_tr = smoothed_tr - smoothed_tr / period + tr[i]
        
        if smoothed_tr > 0:
            plus_di[i] = 100 * smoothed_plus_dm / smoothed_tr
            minus_di[i] = 100 * smoothed_minus_dm / smoothed_tr
        else:
            plus_di[i] = 0
            minus_di[i] = 0
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    sum_dx = np.sum(dx[period:period * 2])
    for i in range(period * 2, n):
        if i == period * 2:
            smoothed_dx = sum_dx
        else:
            smoothed_dx = smoothed_dx - smoothed_dx / period + dx[i]
        
        adx[i] = smoothed_dx / period
    
    return adx


def calculate_volume_avg(volume, period=20):
    """Calculate average volume"""
    n = len(volume)
    avg_vol = np.zeros(n)
    
    for i in range(period - 1, n):
        avg_vol[i] = np.mean(volume[i - period + 1:i + 1])
    
    return avg_vol


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    adx_15m = calculate_adx(high, low, close, period=14)
    vol_avg_15m = calculate_volume_avg(volume, period=20)
    
    # Get 1h HTF data using mtf_data helper
    hma_1h = np.zeros(n)
    adx_1h = np.zeros(n)
    trend_1h = np.zeros(n)
    
    try:
        df_1h = get_htf_data(prices, '1h')
        close_1h = df_1h['close'].values
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        
        hma_1h_raw = calculate_hma(close_1h, period=21)
        adx_1h_raw = calculate_adx(high_1h, low_1h, close_1h, period=14)
        
        hma_1h = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
        adx_1h = align_htf_to_ltf(prices, df_1h, adx_1h_raw)
        
        for i in range(n):
            if hma_1h[i] > 0:
                if close[i] > hma_1h[i]:
                    trend_1h[i] = 1
                elif close[i] < hma_1h[i]:
                    trend_1h[i] = -1
    except Exception:
        pass
    
    # Get 4h HTF data for regime detection
    adx_4h = np.zeros(n)
    
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        adx_4h_raw = calculate_adx(high_4h, low_4h, close_4h, period=14)
        adx_4h = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
    except Exception:
        pass
    
    # Position sizing - DISCRETE levels based on signal confidence
    SIZE_1_SIGNAL = 0.15
    SIZE_2_SIGNALS = 0.25
    SIZE_3_SIGNALS = 0.30
    SIZE_HALF_LONG = 0.125
    SIZE_HALF_SHORT = -0.125
    
    # Thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    ADX_TREND = 25
    ADX_RANGE = 20
    VOL_RATIO_MIN = 0.8
    ATR_STOP_MULT = 2.0
    
    first_valid = 200
    
    # Track position state - use lists then convert
    signals_list = []
    position_side_list = []
    entry_price_list = []
    tp_triggered_list = []
    extreme_price_list = []
    
    for i in range(n):
        signals_list.append(0.0)
        position_side_list.append(0)
        entry_price_list.append(0.0)
        tp_triggered_list.append(0)
        extreme_price_list.append(0.0)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(adx_15m[i]) or atr_15m[i] == 0:
            signals_list[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        adx_val = adx_15m[i]
        adx_4h_val = adx_4h[i]
        hma_1h_val = hma_1h[i]
        vol_ratio = volume[i] / vol_avg_15m[i] if vol_avg_15m[i] > 0 else 0
        
        # Determine regime from 4h ADX
        if adx_4h_val > ADX_TREND:
            regime = "trend"
        elif adx_4h_val < ADX_RANGE:
            regime = "range"
        else:
            regime = "neutral"
        
        # Signal 1: HMA Trend (15m vs 1h alignment)
        signal_hma = 0
        if hma_1h_val > 0:
            if price > hma_15m[i] and price > hma_1h_val:
                signal_hma = 1
            elif price < hma_15m[i] and price < hma_1h_val:
                signal_hma = -1
        
        # Signal 2: RSI Momentum (15m pullback in 1h trend direction)
        signal_rsi = 0
        if trend_1h[i] == 1:
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                signal_rsi = 1
        elif trend_1h[i] == -1:
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                signal_rsi = -1
        
        # Signal 3: Volume Confirmation
        signal_vol = 0
        if vol_ratio >= VOL_RATIO_MIN:
            if trend_1h[i] == 1 and price > hma_15m[i]:
                signal_vol = 1
            elif trend_1h[i] == -1 and price < hma_15m[i]:
                signal_vol = -1
        
        # Ensemble voting
        bullish_signals = sum([1 for s in [signal_hma, signal_rsi, signal_vol] if s == 1])
        bearish_signals = sum([1 for s in [signal_hma, signal_rsi, signal_vol] if s == -1])
        
        # Determine target signal
        target_signal = 0.0
        if bullish_signals >= 2:
            if bullish_signals == 3:
                target_signal = SIZE_3_SIGNALS
            else:
                target_signal = SIZE_2_SIGNALS
        elif bearish_signals >= 2:
            if bearish_signals == 3:
                target_signal = -SIZE_3_SIGNALS
            else:
                target_signal = -SIZE_2_SIGNALS
        elif bullish_signals == 1 or bearish_signals == 1:
            if regime == "trend":
                target_signal = SIZE_1_SIGNAL if bullish_signals == 1 else -SIZE_1_SIGNAL
        
        # Handle existing positions
        prev_side = position_side_list[i - 1]
        
        if prev_side != 0:
            prev_entry = entry_price_list[i - 1]
            if prev_entry == 0:
                prev_entry = close[i - 1]
            prev_tp = tp_triggered_list[i - 1]
            prev_extreme = extreme_price_list[i - 1]
            if prev_extreme == 0:
                prev_extreme = prev_entry
            
            # Update extreme price
            if prev_side == 1:
                current_extreme = max(prev_extreme, price)
            else:
                current_extreme = min(prev_extreme, price)
            
            extreme_price_list[i] = current_extreme
            
            # Stoploss check
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals_list[i] = 0.0
                    position_side_list[i] = 0
                    entry_price_list[i] = 0.0
                    tp_triggered_list[i] = 0
                    extreme_price_list[i] = 0.0
                    continue
                
                # Take profit check
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if prev_tp == 0 and price >= tp_price:
                    signals_list[i] = SIZE_HALF_LONG
                    position_side_list[i] = 1
                    entry_price_list[i] = prev_entry
                    tp_triggered_list[i] = 1
                    extreme_price_list[i] = current_extreme
                    continue
                
                # Trail stop
                if prev_tp == 1:
                    trail_stop = current_extreme - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals_list[i] = 0.0
                        position_side_list[i] = 0
                        entry_price_list[i] = 0.0
                        tp_triggered_list[i] = 0
                        extreme_price_list[i] = 0.0
                        continue
                
                # Hold or exit
                if target_signal > 0:
                    signals_list[i] = target_signal
                    position_side_list[i] = 1
                    entry_price_list[i] = prev_entry
                    tp_triggered_list[i] = prev_tp
                    extreme_price_list[i] = current_extreme
                else:
                    signals_list[i] = 0.0
                    position_side_list[i] = 0
                    entry_price_list[i] = 0.0
                    tp_triggered_list[i] = 0
                    extreme_price_list[i] = 0.0
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals_list[i] = 0.0
                    position_side_list[i] = 0
                    entry_price_list[i] = 0.0
                    tp_triggered_list[i] = 0
                    extreme_price_list[i] = 0.0
                    continue
                
                # Take profit check
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if prev_tp == 0 and price <= tp_price:
                    signals_list[i] = SIZE_HALF_SHORT
                    position_side_list[i] = -1
                    entry_price_list[i] = prev_entry
                    tp_triggered_list[i] = 1
                    extreme_price_list[i] = current_extreme
                    continue
                
                # Trail stop
                if prev_tp == 1:
                    trail_stop = current_extreme + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals_list[i] = 0.0
                        position_side_list[i] = 0
                        entry_price_list[i] = 0.0
                        tp_triggered_list[i] = 0
                        extreme_price_list[i] = 0.0
                        continue
                
                # Hold or exit
                if target_signal < 0:
                    signals_list[i] = target_signal
                    position_side_list[i] = -1
                    entry_price_list[i] = prev_entry
                    tp_triggered_list[i] = prev_tp
                    extreme_price_list[i] = current_extreme
                else:
                    signals_list[i] = 0.0
                    position_side_list[i] = 0
                    entry_price_list[i] = 0.0
                    tp_triggered_list[i] = 0
                    extreme_price_list[i] = 0.0
            continue
        
        # New entry
        if target_signal != 0.0:
            signals_list[i] = target_signal
            position_side_list[i] = 1 if target_signal > 0 else -1
            entry_price_list[i] = price
            tp_triggered_list[i] = 0
            extreme_price_list[i] = price
        else:
            signals_list[i] = 0.0
            position_side_list[i] = 0
    
    return np.array(signals_list)