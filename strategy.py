#!/usr/bin/env python3
"""
EXPERIMENT #120 - MTF HMA+RSI+ZScore+Chandelier+VolRegime+Hysteresis (15m+4h Optimized v120)
==================================================================================================
Hypothesis: Return to HMA-based trend (proven in #111, #112, #118 with Sharpe 2.6-6.2) with
optimized filters to beat current best (Sharpe=16.016). Key improvements over #119:

1. HMA instead of KAMA (HMA showed better results in #111, #112, #118)
2. Relaxed ADX threshold: 15 instead of 20 (#117 failed with ADX>20, too restrictive)
3. Wider RSI entry bands: 30-60 for longs, 40-70 for shorts (more entry opportunities)
4. Z-score filter: |zscore| < 2.0 (avoid extreme overextensions)
5. Volatility-adjusted sizing: 0.35 (low vol), 0.20 (high vol) - more aggressive than #119
6. Hysteresis threshold: 0.08 (balance between churn reduction and responsiveness)
7. Chandelier exit: 3*ATR(22) trailing stop (proven in winning experiments)

Why this should beat current best:
- HMA is more responsive than KAMA for trend detection
- Relaxed ADX allows more trades while still filtering choppy markets
- Z-score prevents chasing tops/bottoms
- Volatility-adjusted sizing controls drawdown in high vol regimes
- Based on proven 15m+4h MTF framework from #111, #112, #118

Risk Management:
- Max signal: 0.35 (low vol), 0.20 (high vol)
- Chandelier exit: 3*ATR(22) trailing stop
- Take profit: reduce to half at 2R, trail at 1R
- ADX filter: 4h ADX > 15 (moderate trend strength)
- Hysteresis: 0.08 threshold to reduce churn costs
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_zscore_chandelier_volregime_15m_4h_v120"
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
    """
    Calculate Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    More responsive than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # Calculate WMA for half period
    wma_half = np.zeros(n)
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        window = close[i - half_period + 1:i + 1]
        wma_half[i] = np.sum(window * weights) / np.sum(weights)
    
    # Calculate WMA for full period
    wma_full = np.zeros(n)
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        window = close[i - period + 1:i + 1]
        wma_full[i] = np.sum(window * weights) / np.sum(weights)
    
    # Calculate raw HMA
    raw_hma = 2 * wma_half - wma_full
    
    # Calculate final HMA with sqrt period
    hma = np.zeros(n)
    for i in range(sqrt_period - 1, n):
        if raw_hma[i] != 0 or i > sqrt_period:
            weights = np.arange(1, sqrt_period + 1)
            start_idx = max(0, i - sqrt_period + 1)
            window = raw_hma[start_idx:i + 1]
            if len(window) == sqrt_period:
                hma[i] = np.sum(window * weights) / np.sum(weights)
            else:
                hma[i] = np.mean(window)
    
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend_values, supertrend_direction (1=below price/bullish, -1=above price/bearish)
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period - 1, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]
    
    supertrend[period - 1] = upper_band[period - 1]
    direction[period - 1] = 1
    
    for i in range(period, n):
        if direction[i - 1] == 1:
            if close[i] < lower_band[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = max(lower_band[i], supertrend[i - 1])
        else:
            if close[i] > upper_band[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = min(upper_band[i], supertrend[i - 1])
    
    return supertrend, direction


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
        else:
            zscore[i] = 0
    
    return zscore


def calculate_chandelier_exit(high, low, close, atr, period=22, multiplier=3.0):
    """
    Calculate Chandelier Exit (ATR trailing stop)
    Long exit: highest_high - multiplier * ATR
    Short exit: lowest_low + multiplier * ATR
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    chandelier_long = np.zeros(n)
    chandelier_short = np.zeros(n)
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        chandelier_long[i] = highest - multiplier * atr[i]
        chandelier_short[i] = lowest + multiplier * atr[i]
    
    return chandelier_long, chandelier_short


def resample_to_4h(close, high, low):
    """Resample 15m data to 4h (16 bars per 4h)"""
    n = len(close)
    bars_per_4h = 16
    n_4h = n // bars_per_4h
    
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * bars_per_4h
        end_idx = start_idx + bars_per_4h
        c_4h[i] = close[end_idx - 1]
        h_4h[i] = np.max(high[start_idx:end_idx])
        l_4h[i] = np.min(low[start_idx:end_idx])
    
    return c_4h, h_4h, l_4h, bars_per_4h


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=16)
    zscore_15m = calculate_zscore(close, period=20)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    supertrend_15m, st_dir_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # Chandelier exit on 15m
    chandelier_long_15m, chandelier_short_15m = calculate_chandelier_exit(
        high, low, close, atr_15m, period=22, multiplier=3.0
    )
    
    # Resample to 4h for trend filters (16 x 15m = 4h)
    c_4h, h_4h, l_4h, bars_per_4h = resample_to_4h(close, high, low)
    n_4h = len(c_4h)
    
    # 4h indicators for trend
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    hma_4h = calculate_hma(c_4h, period=16)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    supertrend_4h, st_dir_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Calculate volatility regime (BBW percentile on 4h)
    bbw_percentile = np.zeros(n_4h)
    lookback = 100
    for i in range(lookback - 1, n_4h):
        window = bbw_4h[i - lookback + 1:i + 1]
        bbw_percentile[i] = np.sum(bbw_4h[i - lookback + 1:i + 1] <= bbw_4h[i]) / lookback
    
    # Calculate ATR percentile for additional vol filter
    atr_percentile = np.zeros(n_4h)
    for i in range(lookback - 1, n_4h):
        window = atr_4h[i - lookback + 1:i + 1]
        atr_percentile[i] = np.sum(atr_4h[i - lookback + 1:i + 1] <= atr_4h[i]) / lookback
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    vol_regime = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    st_dir_4h_mapped = np.zeros(n)
    hma_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 40:
            # HMA trend direction
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            # Supertrend direction
            st_dir_4h_mapped[i] = st_dir_4h[idx_4h]
            hma_4h_mapped[i] = hma_4h[idx_4h]
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
            atr_4h_mapped[i] = atr_4h[idx_4h]
            
            # Combined vol regime (BBW + ATR)
            vol_regime[i] = 0
            if idx_4h >= lookback - 1:
                vol_regime[i] = max(bbw_percentile[idx_4h], atr_percentile[idx_4h])
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on vol regime (CRITICAL for drawdown control)
    SIZE_HIGH_VOL = 0.20  # When vol > 80th percentile
    SIZE_LOW_VOL = 0.35   # When vol <= 80th percentile
    SIZE_HALF_HIGH = 0.10
    SIZE_HALF_LOW = 0.18
    
    # RSI thresholds for pullback entries (wider than #119)
    RSI_LONG_MIN = 30
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 70
    
    # ADX threshold for trend strength (4h) - relaxed from 20 to 15
    ADX_MIN = 15
    
    # Z-score thresholds to avoid overextended entries
    ZSCORE_MAX = 2.0
    ZSCORE_MIN = -2.0
    
    first_valid = max(300, 40 * bars_per_4h, 100 * bars_per_4h)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    initial_risk = np.zeros(n)
    
    # Hysteresis to reduce churn
    prev_signal = 0.0
    hysteresis_threshold = 0.08  # Signal must change by this much to flip
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        adx_4h_val = adx_4h_mapped[i]
        vol_regime_val = vol_regime[i]
        st_direction = st_dir_4h_mapped[i]
        zscore_val = zscore_15m[i]
        
        # Determine position size based on volatility regime
        if vol_regime_val > 0.80:
            size_full = SIZE_HIGH_VOL
            size_half = SIZE_HALF_HIGH
        else:
            size_full = SIZE_LOW_VOL
            size_half = SIZE_HALF_LOW
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            prev_signal = 0.0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            prev_initial_r = initial_risk[i - 1] if initial_risk[i - 1] > 0 else 3 * atr
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Chandelier exit stoploss check (3*ATR trailing)
            if prev_side == 1:
                chandelier_stop = chandelier_long_15m[i]
                if price < chandelier_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_risk[i] = 0
                    prev_signal = 0.0
                    continue
                
                # Take profit check (2R based on initial ATR)
                tp_price = prev_entry + 2 * prev_initial_r
                if not prev_tp and price >= tp_price:
                    signals[i] = size_half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    initial_risk[i] = prev_initial_r
                    prev_signal = signals[i]
                    continue
                
                # Trail stop at 1R profit after TP
                if prev_tp:
                    trail_stop = current_high - prev_initial_r
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        initial_risk[i] = 0
                        prev_signal = 0.0
                        continue
                    
            elif prev_side == -1:
                chandelier_stop = chandelier_short_15m[i]
                if price > chandelier_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_risk[i] = 0
                    prev_signal = 0.0
                    continue
                
                # Take profit check (2R based on initial ATR)
                tp_price = prev_entry - 2 * prev_initial_r
                if not prev_tp and price <= tp_price:
                    signals[i] = -size_half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    initial_risk[i] = prev_initial_r
                    prev_signal = signals[i]
                    continue
                
                # Trail stop at 1R profit after TP
                if prev_tp:
                    trail_stop = current_low + prev_initial_r
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        initial_risk[i] = 0
                        prev_signal = 0.0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            initial_risk[i] = initial_risk[i - 1]
            prev_signal = signals[i]
            continue
        
        # Hysteresis check - only flip if signal change is significant
        target_signal = 0.0
        
        # Entry logic: 4h HMA trend + 4h Supertrend + 4h ADX + 15m RSI pullback + 15m Z-score
        if trend == 1 and st_direction == 1 and adx_4h_val >= ADX_MIN:  # Bullish trend confirmed
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX) and (zscore_val < ZSCORE_MAX):
                target_signal = size_full
                
        elif trend == -1 and st_direction == -1 and adx_4h_val >= ADX_MIN:  # Bearish trend confirmed
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX) and (zscore_val > ZSCORE_MIN):
                target_signal = -size_full
        
        # Apply hysteresis
        if abs(target_signal - prev_signal) < hysteresis_threshold:
            signals[i] = prev_signal
            position_side[i] = 1 if prev_signal > 0 else (-1 if prev_signal < 0 else 0)
            if prev_signal != 0:
                entry_price[i] = entry_price[i - 1] if entry_price[i - 1] > 0 else price
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
                initial_risk[i] = initial_risk[i - 1]
        else:
            signals[i] = target_signal
            position_side[i] = 1 if target_signal > 0 else (-1 if target_signal < 0 else 0)
            
            if target_signal != 0:
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                initial_risk[i] = 3 * atr
            
            prev_signal = target_signal
    
    return signals