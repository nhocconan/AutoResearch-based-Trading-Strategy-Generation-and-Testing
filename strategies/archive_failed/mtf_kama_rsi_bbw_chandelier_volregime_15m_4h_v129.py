#!/usr/bin/env python3
"""
EXPERIMENT #129 - MTF KAMA+RSI+BBW+Chandelier+VolRegime_v129
==================================================================================================
Hypothesis: Beat Sharpe=5.643 (#120) by improving trend detection and regime filtering.
Key insights from history:
1. #120 (Sharpe=5.643): 15m entries + 4h trend + HMA+Supertrend + asymmetric RSI worked best
2. #127: Adding KAMA filter reduced performance (Sharpe=3.508)
3. #128: Simplified back to #120 logic got Sharpe=4.636

New approach for #129:
1. Use KAMA instead of HMA for 4h trend (adaptive to volatility, less whipsaw)
2. Add Bollinger Band Width percentile for regime detection (squeeze=low vol, expansion=high vol)
3. Tighter RSI bands: 35-50 long, 50-65 short (more selective entries)
4. Chandelier multiplier: 2.5*ATR (tighter than 3.0, reduces drawdown)
5. 3-tier position sizing based on BBW percentile (simpler than 4 quartiles)
6. Add MACD histogram momentum confirmation (only enter when momentum aligns)
7. Remove 2-bar confirmation (adds lag, #120 worked without it in earlier versions)
8. Increase hysteresis to 0.20 (reduce churn costs further)

Risk Management:
- Max signal: 0.35 (low vol) down to 0.18 (high vol)
- Chandelier exit: 2.5*ATR(22) trailing stop
- Take profit: 50% position at 2.5R, trail at 1.5R
- leverage=1.0 (position sizing controls risk)

Timeframe: 15m entries with 4h trend filter (proven MTF combination)
"""

import numpy as np
import pandas as pd

name = "mtf_kama_rsi_bbw_chandelier_volregime_15m_4h_v129"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility - moves fast in trends, slow in ranges
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        noise = 0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        er[i] = price_change / noise if noise > 0 else 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
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


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        sma[i] = np.mean(window)
        std = np.std(window)
        upper[i] = sma[i] + std_dev * std
        lower[i] = sma[i] - std_dev * std
        bbw[i] = (upper[i] - lower[i]) / sma[i] if sma[i] > 0 else 0
    
    return sma, upper, lower, bbw


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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    def ema(data, period):
        result = np.zeros(len(data))
        multiplier = 2 / (period + 1)
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    signal_line[slow + signal - 1] = np.mean(macd_line[slow:slow + signal])
    for i in range(slow + signal, n):
        signal_line[i] = (macd_line[i] - signal_line[i - 1]) * (2 / (signal + 1)) + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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
    zscore_15m = calculate_zscore(close, period=20)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    _, _, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # === Resample to 4h for trend filters ===
    c_4h, h_4h, l_4h, bars_per_4h = resample_to_4h(close, high, low)
    n_4h = len(c_4h)
    
    # === 4h indicators for trend direction ===
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
    supertrend_4h, st_dir_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_dev=2.0)
    
    # === BBW Percentile for volatility regime (4h, 100-bar lookback) ===
    bbw_percentile = np.zeros(n_4h)
    lookback = 100
    
    for i in range(lookback - 1, n_4h):
        bbw_window = bbw_4h[i - lookback + 1:i + 1]
        current_bbw = bbw_4h[i]
        bbw_percentile[i] = np.sum(bbw_window <= current_bbw) / lookback
    
    # === Map 4h indicators back to 15m timeframe ===
    trend_4h = np.zeros(n)
    bbw_pct_mapped = np.zeros(n)
    st_dir_4h_mapped = np.zeros(n)
    kama_trend_4h = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 30:
            # Trend: KAMA direction + Supertrend agreement
            kama_trend = 1 if c_4h[idx_4h] > kama_4h[idx_4h] else (-1 if c_4h[idx_4h] < kama_4h[idx_4h] else 0)
            st_trend = st_dir_4h[idx_4h]
            
            # Only count as trend if both agree
            if kama_trend == st_trend and kama_trend != 0:
                trend_4h[i] = kama_trend
            else:
                trend_4h[i] = 0  # No clear trend
            
            kama_trend_4h[i] = kama_trend
            st_dir_4h_mapped[i] = st_trend
            if idx_4h >= lookback - 1:
                bbw_pct_mapped[i] = bbw_percentile[idx_4h]
    
    # === Generate signals with multi-timeframe logic ===
    signals = np.zeros(n)
    
    # Position sizing - 3 DISCRETE levels based on BBW percentile
    SIZE_LOW_VOL = 0.35   # BBW pct < 33% (squeeze, low vol, aggressive)
    SIZE_MED_VOL = 0.25   # BBW pct 33-66%
    SIZE_HIGH_VOL = 0.18  # BBW pct > 66% (expansion, high vol, conservative)
    
    # Tighter asymmetric RSI bands (more selective than #128)
    RSI_LONG_MIN, RSI_LONG_MAX = 35, 50  # Deeper pullback in uptrend
    RSI_SHORT_MIN, RSI_SHORT_MAX = 50, 65  # Stronger rally in downtrend
    
    ZSCORE_MAX = 2.0
    ZSCORE_MIN = -2.0
    HYSTERESIS = 0.20  # Higher than #128 to reduce churn
    MACD_THRESHOLD = 0  # Histogram must be positive for longs, negative for shorts
    
    # Position tracking state
    in_position = False
    position_side = 0
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    initial_risk = 0.0
    prev_signal = 0.0
    
    # Chandelier exit tracking (2.5*ATR, tighter than 3.0)
    chandelier_long = np.zeros(n)
    chandelier_short = np.zeros(n)
    for i in range(22, n):
        highest = np.max(high[i - 22 + 1:i + 1])
        lowest = np.min(low[i - 22 + 1:i + 1])
        chandelier_long[i] = highest - 2.5 * atr_15m[i]
        chandelier_short[i] = lowest + 2.5 * atr_15m[i]
    
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
        vol_val = bbw_pct_mapped[i]
        st_dir = st_dir_4h_mapped[i]
        zscore_val = zscore_15m[i]
        macd_hist = macd_hist_15m[i]
        
        # Determine position size based on volatility regime
        if vol_val < 0.33:
            size_full, size_half = SIZE_LOW_VOL, SIZE_LOW_VOL * 0.5
        elif vol_val < 0.66:
            size_full, size_half = SIZE_MED_VOL, SIZE_MED_VOL * 0.5
        else:
            size_full, size_half = SIZE_HIGH_VOL, SIZE_HIGH_VOL * 0.5
        
        # === Position management (stoploss & take profit) ===
        if in_position:
            # Update extremes
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, price)
                lowest_since_entry = price if lowest_since_entry == 0 else min(lowest_since_entry, price)
            else:
                lowest_since_entry = min(lowest_since_entry, price)
                highest_since_entry = price if highest_since_entry == 0 else max(highest_since_entry, price)
            
            # Chandelier stoploss (2.5*ATR)
            if position_side == 1:
                if price < chandelier_long[i]:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    prev_signal = 0.0
                    continue
                
                # Take profit at 2.5R
                if not tp_triggered and price >= entry_price + 2.5 * initial_risk:
                    signals[i] = size_half
                    tp_triggered = True
                    prev_signal = signals[i]
                    continue
                
                # Trail at 1.5R after TP
                if tp_triggered and price < highest_since_entry - 1.5 * initial_risk:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    prev_signal = 0.0
                    continue
                    
            elif position_side == -1:
                if price > chandelier_short[i]:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    prev_signal = 0.0
                    continue
                
                if not tp_triggered and price <= entry_price - 2.5 * initial_risk:
                    signals[i] = -size_half
                    tp_triggered = True
                    prev_signal = signals[i]
                    continue
                
                if tp_triggered and price > lowest_since_entry + 1.5 * initial_risk:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    prev_signal = 0.0
                    continue
            
            # Hold position
            signals[i] = prev_signal
            continue
        
        # === Entry logic: MTF confirmation with MACD momentum ===
        target_signal = 0.0
        
        # Long: 4h uptrend (KAMA+Supertrend agree) + RSI pullback + Z-score normal + MACD positive
        if trend == 1 and st_dir == 1:
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX) and \
               (ZSCORE_MIN <= zscore_val <= ZSCORE_MAX) and \
               (macd_hist > MACD_THRESHOLD):
                target_signal = size_full
        
        # Short: 4h downtrend (KAMA+Supertrend agree) + RSI pullback + Z-score normal + MACD negative
        elif trend == -1 and st_dir == -1:
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX) and \
               (ZSCORE_MIN <= zscore_val <= ZSCORE_MAX) and \
               (macd_hist < -MACD_THRESHOLD):
                target_signal = -size_full
        
        # === Hysteresis to reduce churn ===
        if abs(target_signal - prev_signal) < HYSTERESIS:
            signals[i] = prev_signal
        else:
            signals[i] = target_signal
            
            if target_signal != 0 and prev_signal == 0:
                # New entry
                in_position = True
                position_side = 1 if target_signal > 0 else -1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
                initial_risk = 2.5 * atr
            elif target_signal == 0 and prev_signal != 0:
                # Exit
                in_position = False
                position_side = 0
            
            prev_signal = target_signal
    
    return signals