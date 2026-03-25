#!/usr/bin/env python3
"""
Experiment #1603: 6h Primary + 1d/1w HTF — Volume-Weighted Momentum + Adaptive Trend

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow). 
Using volume-weighted momentum (VWM) instead of simple ROC captures 
institutional flow better. Combined with KAMA adaptive trend and simple 
HTF bias (1d HMA slope + 1w HMA position), this should generate 30-60 
trades/year with better risk-adjusted returns than pure trend strategies.

Key innovations vs failed 6h attempts:
1. VOLUME-WEIGHTED MOMENTUM: VWM = (close - close[n]) * avg_volume_ratio
   High volume moves get amplified signal weight
2. KAMA EFFICIENCY RATIO: Adjusts trend sensitivity based on market noise
   ER > 0.5 = trending (follow), ER < 0.3 = ranging (mean revert)
3. SIMPLE HTF BIAS: 1d HMA slope direction + 1w HMA position (not complex regime)
4. LOOSE ENTRY THRESHOLDS: VWM cross 0 + volume spike OR KAMA cross + HTF confirm
   Must guarantee ≥30 trades/train (learned from 0-trade failures)
5. ASYMMETRIC SIZING: 0.30 when 1w+1d align, 0.20 when only 1d confirms

Why this should beat mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- Volume weighting captures institutional flow (proven edge in crypto)
- KAMA ER adapts to regime automatically (no CHOP threshold tuning)
- Simpler HTF logic = fewer false filters = more trades
- 6h TF = sweet spot between 4h noise and 12h lag

Entry logic (LOOSE to guarantee trades):
- LONG: 1d_HMA_slope>0 + (VWM>0 crossing up OR KAMA cross above price) + vol>1.2x
- SHORT: 1d_HMA_slope<0 + (VWM<0 crossing down OR KAMA cross below price) + vol>1.2x
- Exit: VWM crosses opposite OR stoploss (2.5x ATR)

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vwm_kama_era_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adjusts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period - 1, n):
        price_change = abs(close[i] - close[i - er_period + 1])
        volatility = np.sum(np.abs(np.diff(close[i - er_period + 1:i + 1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period - 1] = close[er_period - 1]
    
    for i in range(er_period, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama, er

def calculate_vwm(close, volume, period=14):
    """
    Volume-Weighted Momentum
    Momentum amplified by relative volume
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    vwm = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate volume average
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    for i in range(period, n):
        if vol_avg[i] > 0 and vol_avg[i - period] > 0:
            price_momentum = close[i] - close[i - period]
            vol_ratio = volume[i] / vol_avg[i]
            vwm[i] = price_momentum * vol_ratio
    
    return vwm

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_volume_ratio(volume, period=20):
    """Current volume vs average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_21, er_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    vwm_14 = calculate_vwm(close, volume, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_WEAK = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track VWM crossings
    prev_vwm = np.nan
    
    # Warmup period
    min_bars = 80
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_21[i]) or np.isnan(er_10[i]) or np.isnan(vwm_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1d HMA slope + 1w HMA position) ===
        # Calculate 1d HMA slope (current vs 3 bars ago)
        hma_1d_slope = 0.0
        if i >= 3 and not np.isnan(hma_1d_aligned[i-3]):
            hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-3]
        
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLUME-WEIGHTED MOMENTUM SIGNALS ===
        vwm_val = vwm_14[i]
        vwm_prev = prev_vwm if not np.isnan(prev_vwm) else vwm_val
        
        vwm_bull_cross = vwm_val > 0 and vwm_prev <= 0
        vwm_bear_cross = vwm_val < 0 and vwm_prev >= 0
        
        # === KAMA ADAPTIVE TREND ===
        kama_val = kama_21[i]
        er_val = er_10[i]
        
        kama_above_price = kama_val > close[i]
        kama_below_price = kama_val < close[i]
        
        # Efficiency Ratio regime
        is_trending = er_val > 0.5
        is_ranging = er_val < 0.3
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.2 if not np.isnan(vol_ratio[i]) else False
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG entries
        if price_above_1d:  # 1d bias bullish
            # Strong signal: 1w confirms + VWM cross + volume
            if price_above_1w and vwm_bull_cross and vol_confirmed:
                desired_signal = SIZE_STRONG
            # Medium signal: VWM cross OR KAMA cross + volume
            elif (vwm_bull_cross or (kama_below_price and close[i] > kama_val)) and vol_confirmed:
                desired_signal = SIZE_WEAK
            # Weak signal: VWM positive + 1d bullish (catch trends early)
            elif vwm_val > 0 and price_above_1d:
                desired_signal = SIZE_WEAK * 0.5
        
        # SHORT entries
        elif price_below_1d:  # 1d bias bearish
            # Strong signal: 1w confirms + VWM cross + volume
            if price_below_1w and vwm_bear_cross and vol_confirmed:
                desired_signal = -SIZE_STRONG
            # Medium signal: VWM cross OR KAMA cross + volume
            elif (vwm_bear_cross or (kama_above_price and close[i] < kama_val)) and vol_confirmed:
                desired_signal = -SIZE_WEAK
            # Weak signal: VWM negative + 1d bearish
            elif vwm_val < 0 and price_below_1d:
                desired_signal = -SIZE_WEAK * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT SIGNALS (VWM reversal) ===
        if in_position and position_side > 0 and vwm_bear_cross:
            desired_signal = 0.0
        if in_position and position_side < 0 and vwm_bull_cross:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
        elif desired_signal >= SIZE_WEAK * 0.4:
            final_signal = SIZE_WEAK * 0.5
        elif desired_signal <= -SIZE_WEAK * 0.4:
            final_signal = -SIZE_WEAK * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
        prev_vwm = vwm_val
    
    return signals