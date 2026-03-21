#!/usr/bin/env python3
"""
EXPERIMENT #063 - PHASE_ALIGNMENT_CORRELATION_ENSEMBLE_15M_4H_V1
==================================================================================================
Hypothesis: Using Hilbert Transform phase detection combined with signal correlation analysis
will improve entry timing and reduce false signals. When multiple trend indicators align (high
correlation), confidence is higher. Phase detection identifies cycle position for better entries.

Key innovations:
- HILBERT TRANSFORM PHASE: Detects cycle position (0-360°) for timing entries at phase extremes
- SIGNAL CORRELATION MATRIX: Measures agreement between HMA, KAMA, Supertrend, DEMA trends
- PHASE-TREND ALIGNMENT: Only enter when phase suggests continuation AND trend indicators align
- CORRELATION-BASED SIZING: Higher correlation = larger position (more confidence)
- MULTI-TF CONFIRMATION: 4h trend must agree with 15m phase signal

Why this should beat current best (Sharpe=16.016):
- Phase detection provides superior entry timing vs. simple indicator crossovers
- Correlation analysis filters low-confidence setups where indicators disagree
- Combines cycle timing with trend following for optimal risk/reward
- Reduces whipsaw by requiring phase + trend + correlation alignment

Position sizing rules (CRITICAL):
- MAX signal: 0.35 (proven to control drawdown in 2022 crash)
- MIN signal: 0.20 (avoid tiny positions eaten by fees)
- Discrete levels: 0.0, 0.20, 0.28, 0.35 (reduces churn costs)
- Correlation scaling: base_size * (correlation_score / 0.7) capped at 0.35
- Stoploss: 2.5*ATR trailing with 1R trail after 2R profit
"""

import numpy as np
import pandas as pd

name = "phase_alignment_correlation_ensemble_15m_4h_v1"
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


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def calc_wma(data, wma_period):
        result = np.zeros(len(data))
        for i in range(wma_period - 1, len(data)):
            weights = np.arange(1, wma_period + 1)
            window = data[i - wma_period + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma1 = calc_wma(close, half)
    wma2 = calc_wma(close, period)
    raw_hma = 2 * wma1 - wma2
    hma = calc_wma(raw_hma, sqrt_period)
    
    return hma


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    dema = np.zeros(n)
    
    ema1[period - 1] = np.mean(close[:period])
    for i in range(period, n):
        ema1[i] = ema1[i - 1] + (2 / (period + 1)) * (close[i] - ema1[i - 1])
    
    ema2[period - 1] = np.mean(ema1[:period])
    for i in range(period, n):
        ema2[i] = ema2[i - 1] + (2 / (period + 1)) * (ema1[i] - ema2[i - 1])
    
    dema = 2 * ema1 - ema2
    
    return dema


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    direction = np.zeros(n)
    supertrend = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper = mid + multiplier * atr[i]
        lower = mid - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper
            direction[i] = 1
        else:
            if direction[i - 1] == 1:
                if close[i] < upper:
                    supertrend[i] = upper
                    direction[i] = 1
                else:
                    supertrend[i] = lower
                    direction[i] = -1
            else:
                if close[i] > lower:
                    supertrend[i] = lower
                    direction[i] = -1
                else:
                    supertrend[i] = upper
                    direction[i] = 1
    
    return supertrend, direction


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    er = np.zeros(n)
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    sc = np.zeros(n)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_hilbert_phase(close, period=20):
    """
    Calculate Hilbert Transform phase for cycle detection.
    Returns phase angle in degrees (0-360).
    Based on John Ehlers' Hilbert Transform indicators.
    """
    n = len(close)
    if n < period * 2:
        return np.zeros(n), np.zeros(n)
    
    # Detrend the price
    detrend = np.zeros(n)
    for i in range(period, n):
        detrend[i] = close[i] - close[i - period]
    
    # Hilbert Transform - Quadrature component
    i1 = np.zeros(n)  # InPhase
    q1 = np.zeros(n)  # Quadrature
    
    for i in range(6, n):
        i1[i] = detrend[i]
        q1[i] = 0.0962 * detrend[i] + 0.5769 * detrend[i - 2] - 0.5769 * detrend[i - 4] - 0.0962 * detrend[i - 6]
    
    # Smooth I and Q
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    
    for i in range(3, n):
        i2[i] = 0.2 * i1[i] + 0.8 * i2[i - 1] if i > 3 else i1[i]
        q2[i] = 0.2 * q1[i] + 0.8 * q2[i - 1] if i > 3 else q1[i]
    
    # Calculate phase
    phase = np.zeros(n)
    phase_rate = np.zeros(n)
    
    for i in range(2, n):
        if abs(q2[i]) > 0.001:
            phase[i] = np.arctan2(q2[i], i2[i]) * 180 / np.pi
        else:
            phase[i] = phase[i - 1] if i > 2 else 0
        
        phase[i] = phase[i] % 360  # Keep in 0-360 range
        
        if i > 2:
            phase_rate[i] = phase[i] - phase[i - 1]
        else:
            phase_rate[i] = 0
    
    return phase, phase_rate


def calculate_signal_correlation(signals_list, window=20):
    """
    Calculate correlation between multiple signal series.
    Returns average pairwise correlation.
    """
    n = len(signals_list[0])
    correlation = np.zeros(n)
    
    for i in range(window, n):
        corr_sum = 0
        corr_count = 0
        
        for j in range(len(signals_list)):
            for k in range(j + 1, len(signals_list)):
                sig1 = signals_list[j][i - window:i + 1]
                sig2 = signals_list[k][i - window:i + 1]
                
                if np.std(sig1) > 0.001 and np.std(sig2) > 0.001:
                    corr = np.corrcoef(sig1, sig2)[0, 1]
                    if not np.isnan(corr):
                        corr_sum += corr
                        corr_count += 1
        
        if corr_count > 0:
            correlation[i] = corr_sum / corr_count
        else:
            correlation[i] = 0
    
    return correlation


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


def resample_to_higher_tf(close, high, low, volume, bars_per_tf=4):
    """Resample to higher timeframe"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return close.copy(), high.copy(), low.copy(), volume.copy()
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    v_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        if end_idx <= n:
            c_tf[i] = close[end_idx - 1]
            h_tf[i] = np.max(high[start_idx:end_idx])
            l_tf[i] = np.min(low[start_idx:end_idx])
            v_tf[i] = np.sum(volume[start_idx:end_idx])
    
    return c_tf, h_tf, l_tf, v_tf


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=16)
    dema_15m = calculate_dema(close, period=21)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    st_15m, st_dir_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    phase_15m, phase_rate_15m = calculate_hilbert_phase(close, period=20)
    
    # Resample to 4h for trend (16 x 15m = 4h)
    bars_per_4h = 16
    c_4h, h_4h, l_4h, v_4h = resample_to_higher_tf(close, high, low, volume, bars_per_4h)
    
    # 4h indicators for trend
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    hma_4h = calculate_hma(c_4h, period=16)
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    st_4h, st_dir_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    phase_4h, phase_rate_4h = calculate_hilbert_phase(c_4h, period=20)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    kama_trend_4h = np.zeros(n)
    phase_4h_mapped = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    
    n_4h = len(c_4h)
    for i in range(n):
        idx_4h = i // bars_per_4h
        
        if idx_4h < n_4h and idx_4h >= 40:
            # HMA trend
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            # Supertrend direction
            st_trend_4h[i] = st_dir_4h[idx_4h]
            
            # KAMA trend
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                kama_trend_4h[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                kama_trend_4h[i] = -1
            
            # Phase mapped
            phase_4h_mapped[i] = phase_4h[idx_4h] if idx_4h < len(phase_4h) else 0
            atr_4h_mapped[i] = atr_4h[idx_4h] if idx_4h < len(atr_4h) else atr_15m[i]
    
    # Calculate 15m signal series for correlation
    hma_signal_15m = np.where(close > hma_15m, 1, np.where(close < hma_15m, -1, 0))
    dema_signal_15m = np.where(close > dema_15m, 1, np.where(close < dema_15m, -1, 0))
    kama_signal_15m = np.where(close > kama_15m, 1, np.where(close < kama_15m, -1, 0))
    st_signal_15m = st_dir_15m
    
    # Calculate signal correlation (rolling 20-bar window)
    signal_corr = calculate_signal_correlation(
        [hma_signal_15m, dema_signal_15m, kama_signal_15m, st_signal_15m],
        window=20
    )
    
    # Position sizing parameters (DISCRETE levels based on correlation)
    SIZE_LEVELS = {0: 0.20, 1: 0.28, 2: 0.35}
    BASE_SIZE = 0.28
    
    # Phase thresholds for entry timing
    PHASE_LONG_ENTRY = 45  # Enter long when phase crosses up through 45°
    PHASE_SHORT_ENTRY = 225  # Enter short when phase crosses down through 225°
    PHASE_LONG_EXIT = 135  # Exit long when phase crosses above 135°
    PHASE_SHORT_EXIT = 315  # Exit short when phase crosses below 315°
    
    # Correlation thresholds
    CORR_HIGH = 0.7  # High correlation = high confidence
    CORR_MEDIUM = 0.4  # Medium correlation = medium confidence
    CORR_LOW = 0.2  # Low correlation = low confidence (reduce size)
    
    # Stoploss multipliers
    ATR_STOP_NORMAL = 2.5
    ATR_STOP_HIGH_VOL = 3.5
    
    first_valid = max(200, 40 * bars_per_4h + 100)
    
    # Generate signals with phase alignment and correlation confidence
    signals = np.zeros(n)
    
    # Track position state for stoploss/TP
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    last_signal = np.zeros(n)
    prev_phase = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            last_signal[i] = signals[i-1] if i > 0 else 0
            prev_phase[i] = prev_phase[i-1] if i > 0 else 0
            continue
        
        # 4h regime signals
        hma_trend = trend_4h[i]
        st_trend = st_trend_4h[i]
        kama_trend = kama_trend_4h[i]
        atr_4h_val = atr_4h_mapped[i]
        phase_4h_val = phase_4h_mapped[i]
        
        # 15m entry signals
        price = close[i]
        hma_15m_val = hma_15m[i]
        dema_15m_val = dema_15m[i]
        kama_15m_val = kama_15m[i]
        st_dir = st_dir_15m[i]
        rsi_val = rsi_15m[i]
        atr_15m_val = atr_15m[i]
        phase_15m_val = phase_15m[i]
        phase_rate = phase_rate_15m[i]
        corr_val = signal_corr[i]
        
        # Store previous phase for crossing detection
        prev_phase[i] = prev_phase[i - 1] if i > 0 else phase_15m_val
        
        # Determine volatility regime and adaptive ATR stop
        vol_ratio = atr_15m_val / max(atr_4h_val / 4, atr_15m_val * 0.5)
        if vol_ratio > 1.5:
            vol_regime = "high"
            atr_stop_mult = ATR_STOP_HIGH_VOL
            vol_size_mult = 0.7
        elif vol_ratio < 0.7:
            vol_regime = "low"
            atr_stop_mult = ATR_STOP_NORMAL
            vol_size_mult = 1.2
        else:
            vol_regime = "normal"
            atr_stop_mult = ATR_STOP_NORMAL
            vol_size_mult = 1.0
        
        # PHASE-BASED ENTRY SIGNALS
        # Phase 0-90: Accumulation (prepare long)
        # Phase 90-180: Markup (hold long)
        # Phase 180-270: Distribution (prepare short)
        # Phase 270-360: Markdown (hold short)
        
        phase_signal = 0
        phase_cross_up = phase_15m_val > PHASE_LONG_ENTRY and prev_phase[i] <= PHASE_LONG_ENTRY
        phase_cross_down = phase_15m_val < PHASE_SHORT_ENTRY and prev_phase[i] >= PHASE_SHORT_EXIT
        
        # Phase momentum (positive = bullish, negative = bearish)
        phase_momentum = phase_rate > 0
        
        # Generate phase-based signal
        if phase_cross_up and phase_momentum:
            phase_signal = 1  # Long entry signal
        elif phase_cross_down and not phase_momentum:
            phase_signal = -1  # Short entry signal
        elif phase_15m_val > PHASE_LONG_EXIT:
            phase_signal = 0  # Exit long zone
        elif phase_15m_val < 45 or phase_15m_val > 315:
            phase_signal = 0  # Neutral zone
        
        # CORRELATION-BASED CONFIDENCE
        if corr_val >= CORR_HIGH:
            corr_level = 2  # High confidence
            corr_size_mult = 1.0
        elif corr_val >= CORR_MEDIUM:
            corr_level = 1  # Medium confidence
            corr_size_mult = 0.8
        else:
            corr_level = 0  # Low confidence
            corr_size_mult = 0.6
        
        # TREND ALIGNMENT (4h must agree with 15m phase signal)
        trend_aligned = False
        if phase_signal == 1 and hma_trend >= 0 and st_trend >= 0:
            trend_aligned = True
        elif phase_signal == -1 and hma_trend <= 0 and st_trend <= 0:
            trend_aligned = True
        elif phase_signal == 0:
            trend_aligned = True  # No position, alignment not needed
        
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
            
            # Stoploss check
            if prev_side == 1:
                stoploss_price = prev_entry - atr_stop_mult * atr_15m_val
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    last_signal[i] = 0.0
                    prev_phase[i] = phase_15m_val
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * atr_stop_mult * atr_15m_val
                if not prev_tp and price >= tp_price:
                    prev_signal_val = signals[i - 1]
                    signals[i] = prev_side * 0.5 * abs(prev_signal_val) if prev_signal_val != 0 else 0.0
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    last_signal[i] = signals[i]
                    prev_phase[i] = phase_15m_val
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - atr_stop_mult * atr_15m_val
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        last_signal[i] = 0.0
                        prev_phase[i] = phase_15m_val
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + atr_stop_mult * atr_15m_val
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    last_signal[i] = 0.0
                    prev_phase[i] = phase_15m_val
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * atr_stop_mult * atr_15m_val
                if not prev_tp and price <= tp_price:
                    prev_signal_val = signals[i - 1]
                    signals[i] = prev_side * 0.5 * abs(prev_signal_val) if prev_signal_val != 0 else 0.0
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    last_signal[i] = signals[i]
                    prev_phase[i] = phase_15m_val
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + atr_stop_mult * atr_15m_val
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        last_signal[i] = 0.0
                        prev_phase[i] = phase_15m_val
                        continue
            
            # Maintain position if phase and trend still agree
            maintain_signal = False
            if prev_side == 1:
                # Maintain long if phase in markup zone (90-180) and trend bullish
                if 90 <= phase_15m_val <= 180 and hma_trend >= 0:
                    maintain_signal = True
            elif prev_side == -1:
                # Maintain short if phase in markdown zone (270-360) and trend bearish
                if 270 <= phase_15m_val <= 360 and hma_trend <= 0:
                    maintain_signal = True
            
            if maintain_signal and trend_aligned:
                target_size = SIZE_LEVELS.get(corr_level, 0.20)
                target_size = max(min(target_size, 0.35), 0.20)
                target_size = target_size * vol_size_mult * corr_size_mult
                target_size = max(min(target_size, 0.35), 0.20)
                
                prev_size = abs(signals[i - 1])
                # Hysteresis: don't reduce size unless correlation drops significantly
                if corr_level >= 1 or target_size >= prev_size - 0.05:
                    signals[i] = prev_side * target_size
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
            else:
                # Phase suggests exit
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            prev_phase[i] = phase_15m_val
            continue
        
        # Entry logic with phase alignment, trend confirmation, and correlation confidence
        # Require: phase signal + trend alignment + minimum correlation
        
        entry_threshold_corr = 1  # At least medium correlation
        
        if phase_signal == 1 and trend_aligned and corr_level >= entry_threshold_corr:
            target_size = SIZE_LEVELS.get(corr_level, 0.20)
            target_size = max(min(target_size, 0.35), 0.20)
            target_size = target_size * vol_size_mult * corr_size_mult
            target_size = max(min(target_size, 0.35), 0.20)
            
            signals[i] = target_size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            
        elif phase_signal == -1 and trend_aligned and corr_level >= entry_threshold_corr:
            target_size = SIZE_LEVELS.get(corr_level, 0.20)
            target_size = max(min(target_size, 0.35), 0.20)
            target_size = target_size * vol_size_mult * corr_size_mult
            target_size = max(min(target_size, 0.35), 0.20)
            
            signals[i] = -target_size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
        
        last_signal[i] = signals[i]
        prev_phase[i] = phase_15m_val
    
    return signals