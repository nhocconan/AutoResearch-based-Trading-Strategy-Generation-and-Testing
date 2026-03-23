# Strategy: adaptive_kama_rsi_regime_1h_4h_v1

## Status
ACTIVE - Sharpe=0.280 | Return=+60.2% | DD=-33.8%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.100 | +8.0% | -42.5% | 1751 |
| ETHUSDT | 0.293 | +43.0% | -27.6% | 1838 |
| SOLUSDT | 0.646 | +129.6% | -31.3% | 1783 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -1.570 | -16.1% | -21.4% | 559 |
| ETHUSDT | 0.121 | +6.6% | -15.0% | 524 |
| SOLUSDT | 0.766 | +26.0% | -17.7% | 506 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #077 - ADAPTIVE_KAMA_RSI_REGIME_1H_4H_V1
==================================================================================================
Hypothesis: Simplify the ensemble approach by using cleaner regime detection (ADX + BBW combined)
with adaptive signal selection. Use KAMA for trend (adapts to volatility) + RSI for momentum.
1h entries with 4h trend filter (proven in #072 with Sharpe=0.589).

Key improvements from failed experiments:
- Simpler signal logic (2 signals instead of 3) - reduces noise and overfitting
- Combined regime detection (ADX > 25 AND BBW percentile < 0.4 = trend regime)
- KAMA adapts smoothing based on market efficiency ratio (better than HMA in chop)
- RSI with dynamic thresholds based on regime (stricter in trend, looser in mean-reversion)
- Conservative sizing: 0.25 base, 0.35 max on high confidence
- Clear stoploss/takeprofit logic with proper state tracking

Why this should work:
- #072 showed 1h/4h MTF works well (Sharpe=0.589)
- KAMA outperformed HMA in #070 (Sharpe=1.256)
- Simpler logic reduces overfitting risk from #076 crash
- Proper variable scoping avoids 'close' and 'prev_side' errors
"""

import numpy as np
import pandas as pd

name = "adaptive_kama_rsi_regime_1h_4h_v1"
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
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
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


def resample_to_timeframe(close, high, low, bars_per_tf):
    """Resample data to higher timeframe"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return np.zeros(1), np.zeros(1), np.zeros(1)
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        c_tf[i] = close[end_idx - 1]
        h_tf[i] = np.max(high[start_idx:end_idx])
        l_tf[i] = np.min(low[start_idx:end_idx])
    
    return c_tf, h_tf, l_tf


def map_tf_to_base(tf_array, bars_per_tf, base_length):
    """Map higher timeframe array back to base timeframe"""
    mapped = np.zeros(base_length)
    n_tf = len(tf_array)
    
    for i in range(base_length):
        tf_idx = i // bars_per_tf
        if tf_idx < n_tf:
            mapped[i] = tf_array[tf_idx]
    
    return mapped


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Position sizing constants
    SIZE_BASE = 0.25
    SIZE_HIGH = 0.35
    SIZE_LOW = 0.15
    
    # Regime thresholds
    ADX_TREND_THRESHOLD = 25
    BBW_TREND_PERCENTILE = 0.40  # Below 40th percentile = low vol (trend)
    BBW_MR_PERCENTILE = 0.70  # Above 70th percentile = high vol (mean reversion)
    
    # RSI thresholds by regime
    RSI_TREND_LONG = 55
    RSI_TREND_SHORT = 45
    RSI_MR_LONG = 35
    RSI_MR_SHORT = 65
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Timeframe conversion: 4h = 4 x 1h
    bars_per_4h = 4
    
    # Base timeframe (1h) indicators
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx_1h = calculate_adx(high, low, close, period=14)
    bb_upper_1h, bb_mid_1h, bb_lower_1h, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    supertrend_1h, st_dir_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # 4h timeframe indicators for trend filter
    c_4h, h_4h, l_4h = resample_to_timeframe(close, high, low, bars_per_4h)
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    
    # Map 4h indicators to 1h
    kama_4h_mapped = map_tf_to_base(kama_4h, bars_per_4h, n)
    adx_4h_mapped = map_tf_to_base(adx_4h, bars_per_4h, n)
    bbw_pct_4h_mapped = map_tf_to_base(bbw_pct_4h, bars_per_4h, n)
    
    # Minimum warmup period
    first_valid = max(200, 100 * bars_per_4h, 28, 45)
    
    # Initialize output arrays
    signals = np.zeros(n)
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for invalid data
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        rsi_val = rsi_1h[i]
        adx_4h_val = adx_4h_mapped[i]
        bbw_pct = bbw_pct_4h_mapped[i]
        
        # === REGIME DETECTION ===
        # Trend regime: ADX > 25 AND BBW percentile < 0.40
        # Mean reversion regime: BBW percentile > 0.70
        # Neutral: everything else
        is_trend_regime = (adx_4h_val > ADX_TREND_THRESHOLD) and (bbw_pct < BBW_TREND_PERCENTILE)
        is_mr_regime = bbw_pct > BBW_MR_PERCENTILE
        is_neutral = not is_trend_regime and not is_mr_regime
        
        # === 4H TREND FILTER ===
        kama_4h_idx = i // bars_per_4h
        if kama_4h_idx < len(kama_4h):
            trend_4h = 1 if c_4h[kama_4h_idx] > kama_4h[kama_4h_idx] else -1
        else:
            trend_4h = 0
        
        # === 1H SIGNALS ===
        # KAMA trend signal
        kama_signal_1h = 1 if close[i] > kama_1h[i] else -1
        
        # Supertrend signal
        st_signal_1h = st_dir_1h[i]
        
        # RSI momentum signal
        if is_trend_regime:
            rsi_signal = 1 if rsi_val > RSI_TREND_LONG else (-1 if rsi_val < RSI_TREND_SHORT else 0)
        elif is_mr_regime:
            rsi_signal = 1 if rsi_val < RSI_MR_LONG else (-1 if rsi_val > RSI_MR_SHORT else 0)
        else:
            rsi_signal = 1 if rsi_val > 50 else (-1 if rsi_val < 50 else 0)
        
        # Bollinger position signal
        bb_signal = 1 if price < bb_lower_1h[i] else (-1 if price > bb_upper_1h[i] else 0)
        
        # === COMBINED SIGNAL LOGIC ===
        final_signal = 0
        signal_confidence = 0
        
        if is_trend_regime:
            # Trend following: require 4h trend alignment + 1h confirmation
            if trend_4h == 1:
                # Long setup
                trend_votes = 0
                if kama_signal_1h == 1:
                    trend_votes += 1
                if st_signal_1h == 1:
                    trend_votes += 1
                if rsi_signal == 1:
                    trend_votes += 1
                
                if trend_votes >= 2:
                    final_signal = 1
                    signal_confidence = trend_votes / 3.0
            
            elif trend_4h == -1:
                # Short setup
                trend_votes = 0
                if kama_signal_1h == -1:
                    trend_votes += 1
                if st_signal_1h == -1:
                    trend_votes += 1
                if rsi_signal == -1:
                    trend_votes += 1
                
                if trend_votes >= 2:
                    final_signal = -1
                    signal_confidence = trend_votes / 3.0
        
        elif is_mr_regime:
            # Mean reversion: fade extremes
            mr_votes_long = 0
            if bb_signal == 1:
                mr_votes_long += 1
            if rsi_signal == 1:
                mr_votes_long += 1
            if kama_signal_1h == -1 and price < kama_1h[i] - atr:
                mr_votes_long += 1
            
            mr_votes_short = 0
            if bb_signal == -1:
                mr_votes_short += 1
            if rsi_signal == -1:
                mr_votes_short += 1
            if kama_signal_1h == 1 and price > kama_1h[i] + atr:
                mr_votes_short += 1
            
            if mr_votes_long >= 2:
                final_signal = 1
                signal_confidence = mr_votes_long / 3.0
            elif mr_votes_short >= 2:
                final_signal = -1
                signal_confidence = mr_votes_short / 3.0
        
        else:
            # Neutral regime: require strong agreement
            neutral_votes = 0
            if kama_signal_1h == 1 and trend_4h == 1:
                neutral_votes += 1
            if st_signal_1h == 1:
                neutral_votes += 1
            if rsi_signal == 1:
                neutral_votes += 1
            
            if neutral_votes >= 2:
                final_signal = 1
                signal_confidence = neutral_votes / 3.0
            else:
                neutral_votes_neg = 0
                if kama_signal_1h == -1 and trend_4h == -1:
                    neutral_votes_neg += 1
                if st_signal_1h == -1:
                    neutral_votes_neg += 1
                if rsi_signal == -1:
                    neutral_votes_neg += 1
                
                if neutral_votes_neg >= 2:
                    final_signal = -1
                    signal_confidence = neutral_votes_neg / 3.0
        
        # === POSITION MANAGEMENT ===
        prev_side = position_side[i - 1]
        
        if prev_side != 0:
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
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE * 0.5
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit after TP
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE * 0.5
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit after TP
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered and signal agrees
            if (prev_side == 1 and final_signal >= 0) or (prev_side == -1 and final_signal <= 0):
                signals[i] = signals[i - 1]
                position_side[i] = prev_side
                entry_price[i] = prev_entry
                tp_triggered[i] = prev_tp
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
            else:
                # Exit if signal reverses
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            continue
        
        # === ENTRY LOGIC ===
        if final_signal == 1 and signal_confidence > 0.5:
            if signal_confidence > 0.75:
                signals[i] = SIZE_HIGH
            else:
                signals[i] = SIZE_BASE
            
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif final_signal == -1 and signal_confidence > 0.5:
            if signal_confidence > 0.75:
                signals[i] = -SIZE_HIGH
            else:
                signals[i] = -SIZE_BASE
            
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 10:20
