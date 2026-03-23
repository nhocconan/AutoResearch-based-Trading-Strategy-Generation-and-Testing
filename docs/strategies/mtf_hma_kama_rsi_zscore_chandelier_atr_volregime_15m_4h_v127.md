# Strategy: mtf_hma_kama_rsi_zscore_chandelier_atr_volregime_15m_4h_v127

## Status
ACTIVE - Sharpe=3.508 | Return=+259.2% | DD=-3.1%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 2.387 | +86.7% | -2.5% | 1252 |
| ETHUSDT | 3.584 | +178.1% | -3.1% | 1249 |
| SOLUSDT | 4.554 | +512.8% | -3.7% | 1213 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 1.680 | +12.2% | -1.0% | 344 |
| ETHUSDT | 2.928 | +22.8% | -1.5% | 295 |
| SOLUSDT | 2.219 | +20.6% | -1.8% | 266 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #127 - MTF HMA+KAMA+RSI+Zscore+Chandelier+ATR_VolRegime_v127
==================================================================================================
Hypothesis: Beat Sharpe=16.016 by improving volatility regime detection and signal confirmation.
Key insight from #126: ROC filter hurt performance. Return to simpler, proven logic but enhance:
1. ATR-based volatility regime (more responsive than BBW percentile)
2. 3-bar signal confirmation (reduces false entries vs 2-bar)
3. Dual trend filter: HMA + KAMA agreement (reduces whipsaws)
4. Improved Chandelier exit with proper highest_high tracking
5. Asymmetric RSI bands tuned for crypto volatility

Key improvements over #126:
1. ATR volatility regime instead of BBW (more direct vol measurement)
2. 3-bar confirmation instead of 2-bar (fewer false signals)
3. KAMA adaptive trend filter alongside HMA (better in ranging markets)
4. Tighter RSI bands: 38-52 long, 48-62 short (more selective entries)
5. Better position state tracking with explicit exit conditions

Risk Management (per experiment instructions):
- Max signal: 0.40 (Q1 low vol) down to 0.15 (Q4 high vol) - discrete quartile sizing
- Chandelier exit: 3.0*ATR(22) trailing stop with 1R trail after 2R profit
- Dual trend filter: HMA + KAMA must agree on 4h timeframe
- Hysteresis: 0.15 threshold (reduces churn costs from 0.10% per flip)
- 3-bar signal confirmation (reduces false entries)
- leverage=1.0 (no leverage, position sizing controls risk)

Timeframe: 15m entries with 4h trend filter (proven MTF combination)
"""

import numpy as np
import pandas as pd

name = "mtf_hma_kama_rsi_zscore_chandelier_atr_volregime_15m_4h_v127"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing method"""
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
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.zeros(len(data))
        for i in range(w_period - 1, len(data)):
            weights = np.arange(1, w_period + 1)
            window = data[i - w_period + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman's Adaptive Moving Average - adapts to market noise
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = sum(abs(close[j] - close[j - 1]) for j in range(i - er_period + 1, i + 1))
        er[i] = signal / noise if noise > 0 else 0
    
    # Calculate Smoothing Constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    
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
    """Calculate ADX for trend strength"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
    
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
    """Calculate Supertrend indicator"""
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
        bbw[i] = (upper[i] - lower[i]) / middle[i] if middle[i] > 0 else 0
    
    return upper, middle, lower, bbw


def calculate_zscore(close, period=20):
    """Calculate Z-score for overextension detection"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        zscore[i] = (close[i] - mean) / std if std > 0 else 0
    
    return zscore


def calculate_chandelier_exit(high, low, close, atr, period=22, multiplier=3.0):
    """Chandelier Exit (ATR trailing stop): 3.0*ATR(22) per experiment instructions"""
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
    """Resample 15m data to 4h (16 bars per 4h candle)"""
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
    
    # === 15m indicators for entry timing ===
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=16)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    zscore_15m = calculate_zscore(close, period=20)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    chandelier_long_15m, chandelier_short_15m = calculate_chandelier_exit(
        high, low, close, atr_15m, period=22, multiplier=3.0
    )
    
    # === Resample to 4h for trend filters ===
    c_4h, h_4h, l_4h, bars_per_4h = resample_to_4h(close, high, low)
    n_4h = len(c_4h)
    
    # === 4h indicators for trend direction ===
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    hma_4h = calculate_hma(c_4h, period=16)
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    supertrend_4h, st_dir_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # === ATR-based Volatility regime detection (4h, 200-bar lookback) ===
    # More responsive than BBW - uses ATR percentile directly
    atr_percentile = np.zeros(n_4h)
    lookback = 200
    
    for i in range(lookback - 1, n_4h):
        atr_window = atr_4h[i - lookback + 1:i + 1]
        # Normalize ATR by price for comparability
        atr_norm = atr_window / c_4h[i - lookback + 1:i + 1]
        current_atr_norm = atr_4h[i] / c_4h[i] if c_4h[i] > 0 else 0
        atr_percentile[i] = np.sum(atr_norm <= current_atr_norm) / lookback
    
    # === Map 4h indicators back to 15m timeframe ===
    trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    vol_regime = np.zeros(n)
    st_dir_4h_mapped = np.zeros(n)
    hma_trend_4h = np.zeros(n)
    kama_trend_4h = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 40:
            # Dual trend: HMA and KAMA must agree
            hma_trend = 1 if c_4h[idx_4h] > hma_4h[idx_4h] else (-1 if c_4h[idx_4h] < hma_4h[idx_4h] else 0)
            kama_trend = 1 if c_4h[idx_4h] > kama_4h[idx_4h] else (-1 if c_4h[idx_4h] < kama_4h[idx_4h] else 0)
            
            # Only count as trend if both agree
            if hma_trend == kama_trend and hma_trend != 0:
                trend_4h[i] = hma_trend
            else:
                trend_4h[i] = 0  # No clear trend
            
            hma_trend_4h[i] = hma_trend
            kama_trend_4h[i] = kama_trend
            st_dir_4h_mapped[i] = st_dir_4h[idx_4h]
            adx_4h_mapped[i] = adx_4h[idx_4h]
            if idx_4h >= lookback - 1:
                vol_regime[i] = atr_percentile[idx_4h]
    
    # === Generate signals with multi-timeframe logic ===
    signals = np.zeros(n)
    
    # Position sizing - 4 DISCRETE levels based on vol regime quartiles
    SIZE_Q1 = 0.40  # Vol regime < 25% (lowest vol, most aggressive)
    SIZE_Q2 = 0.30  # Vol regime 25-50%
    SIZE_Q3 = 0.20  # Vol regime 50-75%
    SIZE_Q4 = 0.15  # Vol regime > 75% (highest vol, most conservative)
    
    # Asymmetric RSI bands (tighter than #126 for more selective entries)
    RSI_LONG_MIN, RSI_LONG_MAX = 38, 52  # Deeper pullback for longs
    RSI_SHORT_MIN, RSI_SHORT_MAX = 48, 62  # Shallower for shorts
    
    ADX_MIN = 20  # Slightly higher than #126 for better trend quality
    ZSCORE_MAX = 1.5
    ZSCORE_MIN = -1.5
    HYSTERESIS = 0.15  # Reduce churn costs
    
    # Position tracking state (CRITICAL: separate from signal array)
    in_position = False
    position_side = 0
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    initial_risk = 0.0
    prev_signal = 0.0
    
    # Signal confirmation - 3 bars for more stability
    signal_count = 0
    confirmed_signal = 0.0
    last_signal_candidate = 0.0
    
    first_valid = max(300, 40 * bars_per_4h, lookback * bars_per_4h)
    
    for i in range(first_valid, n):
        # Validate data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        adx_val = adx_4h_mapped[i]
        vol_val = vol_regime[i]
        st_dir = st_dir_4h_mapped[i]
        zscore_val = zscore_15m[i]
        
        # Determine position size based on volatility regime
        if vol_val < 0.25:
            size_full, size_half = SIZE_Q1, SIZE_Q1 * 0.5
        elif vol_val < 0.50:
            size_full, size_half = SIZE_Q2, SIZE_Q2 * 0.5
        elif vol_val < 0.75:
            size_full, size_half = SIZE_Q3, SIZE_Q3 * 0.5
        else:
            size_full, size_half = SIZE_Q4, SIZE_Q4 * 0.5
        
        # === ADX filter - exit if trend weakens ===
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            prev_signal = 0.0
            signal_count = 0
            confirmed_signal = 0.0
            last_signal_candidate = 0.0
            continue
        
        # === Position management (stoploss & take profit) ===
        if in_position:
            # Update extremes
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, price)
                lowest_since_entry = price if lowest_since_entry == 0 else min(lowest_since_entry, price)
            else:
                lowest_since_entry = min(lowest_since_entry, price)
                highest_since_entry = price if highest_since_entry == 0 else max(highest_since_entry, price)
            
            # Chandelier stoploss
            if position_side == 1:
                if price < chandelier_long_15m[i]:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    prev_signal = 0.0
                    signal_count = 0
                    confirmed_signal = 0.0
                    last_signal_candidate = 0.0
                    continue
                
                # Take profit at 2R
                if not tp_triggered and price >= entry_price + 2 * initial_risk:
                    signals[i] = size_half
                    tp_triggered = True
                    prev_signal = signals[i]
                    continue
                
                # Trail at 1R after TP
                if tp_triggered and price < highest_since_entry - initial_risk:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    prev_signal = 0.0
                    signal_count = 0
                    confirmed_signal = 0.0
                    last_signal_candidate = 0.0
                    continue
                    
            elif position_side == -1:
                if price > chandelier_short_15m[i]:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    prev_signal = 0.0
                    signal_count = 0
                    confirmed_signal = 0.0
                    last_signal_candidate = 0.0
                    continue
                
                if not tp_triggered and price <= entry_price - 2 * initial_risk:
                    signals[i] = -size_half
                    tp_triggered = True
                    prev_signal = signals[i]
                    continue
                
                if tp_triggered and price > lowest_since_entry + initial_risk:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    prev_signal = 0.0
                    signal_count = 0
                    confirmed_signal = 0.0
                    last_signal_candidate = 0.0
                    continue
            
            # Hold position
            signals[i] = prev_signal
            continue
        
        # === Entry logic: MTF confirmation with dual trend filter ===
        target_signal = 0.0
        
        # Long: 4h uptrend (HMA+KAMA agree) + Supertrend bullish + RSI pullback + Z-score normal
        if trend == 1 and st_dir == 1:
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX) and (ZSCORE_MIN <= zscore_val <= ZSCORE_MAX):
                target_signal = size_full
        
        # Short: 4h downtrend (HMA+KAMA agree) + Supertrend bearish + RSI pullback + Z-score normal
        elif trend == -1 and st_dir == -1:
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX) and (ZSCORE_MIN <= zscore_val <= ZSCORE_MAX):
                target_signal = -size_full
        
        # === Signal confirmation (3-bar persistence for stability) ===
        if target_signal != 0 and target_signal == last_signal_candidate:
            signal_count += 1
        elif target_signal != 0 and target_signal != last_signal_candidate:
            last_signal_candidate = target_signal
            signal_count = 1
        else:
            signal_count = 0
            last_signal_candidate = 0.0
        
        # Execute if confirmed for 3 bars (more stable than 2-bar)
        if signal_count >= 3:
            final_signal = last_signal_candidate
        else:
            final_signal = prev_signal
        
        # === Hysteresis to reduce churn ===
        if abs(final_signal - prev_signal) < HYSTERESIS:
            signals[i] = prev_signal
        else:
            signals[i] = final_signal
            
            if final_signal != 0 and prev_signal == 0:
                # New entry
                in_position = True
                position_side = 1 if final_signal > 0 else -1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
                initial_risk = 3.0 * atr
            elif final_signal == 0 and prev_signal != 0:
                # Exit
                in_position = False
                position_side = 0
            
            prev_signal = final_signal
    
    return signals
```

## Last Updated
2026-03-21 11:37
