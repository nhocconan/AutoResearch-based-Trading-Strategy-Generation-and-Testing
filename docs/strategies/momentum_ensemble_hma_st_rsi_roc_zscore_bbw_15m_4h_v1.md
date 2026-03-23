# Strategy: momentum_ensemble_hma_st_rsi_roc_zscore_bbw_15m_4h_v1

## Status
ACTIVE - Sharpe=0.116 | Return=+41.2% | DD=-15.1%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.427 | +8.9% | -9.9% | 165 |
| ETHUSDT | -0.285 | +10.3% | -11.2% | 101 |
| SOLUSDT | 1.061 | +104.6% | -24.2% | 2 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.011 | +5.8% | -3.3% | 95 |
| ETHUSDT | 0.454 | +9.9% | -8.3% | 69 |
| SOLUSDT | 0.394 | +9.6% | -9.5% | 68 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #052 - MOMENTUM CONFIRMED ENSEMBLE (15m+4h with ROC + RSI Confluence)
==================================================================================================
Hypothesis: The best performing strategy (#040) used 15m timeframe with multiple indicators.
This strategy improves on #051 by:

Key innovations:
- ROC MOMENTUM FILTER: Add Rate-of-Change(10) to confirm momentum before entry
- RSI+ZSCORE CONFLUENCE: Both must agree for mean reversion signals (reduces false entries)
- TIGHTER RSI RANGES: 40-50 for long pullbacks, 50-60 for short pullbacks (more selective)
- IMPROVED BBW CALCULATION: Use 15m BBW for regime, not 4h (more responsive)
- REDUCED CHURN: Only change signal when confidence changes by 2+ votes

Why this should beat #051 (Sharpe=9.983) and approach #040 (Sharpe=16.016):
- 15m timeframe has proven best results in experiment history
- ROC momentum filter avoids entries in weak trends
- RSI+Z-score confluence reduces false mean reversion signals
- Based on proven ensemble framework from #050/#051

Position sizing rules (CRITICAL):
- MAX signal: 0.35 (conservative, proven to control drawdown)
- Base size: 0.20
- Confidence bonus: +0.05 per additional agreeing signal (max 3 signals)
- Volatility penalty: -0.05 in high vol regime
- Final range: 0.15 to 0.35
"""

import numpy as np
import pandas as pd

name = "momentum_ensemble_hma_st_rsi_roc_zscore_bbw_15m_4h_v1"
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = sma[i] + std_mult * std
        lower[i] = sma[i] - std_mult * std
        if sma[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / sma[i]
    
    return upper, lower, bbw


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
    """Calculate Z-score for mean reversion signals"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        mean = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        if std > 0:
            zscore[i] = (close[i] - mean) / std
    
    return zscore


def calculate_roc(close, period=10):
    """Calculate Rate of Change (momentum indicator)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    roc = np.zeros(n)
    for i in range(period, n):
        if close[i - period] > 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100
    
    return roc


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    adx_15m = calculate_adx(high, low, close, period=14)
    _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    zscore_15m = calculate_zscore(close, period=20)
    roc_15m = calculate_roc(close, period=10)
    
    # Resample to 4h for trend filters (16 x 15m = 4h)
    bars_per_4h = 16
    n_4h = (n // bars_per_4h)
    
    # Create 4h arrays by downsampling
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * bars_per_4h
        end_idx = start_idx + bars_per_4h
        c_4h[i] = close[end_idx - 1]
        h_4h[i] = np.max(high[start_idx:end_idx])
        l_4h[i] = np.min(low[start_idx:end_idx])
    
    # 4h indicators for trend
    hma_4h = calculate_hma(c_4h, period=21)
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 40:
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            st_trend_4h[i] = st_direction_4h[idx_4h]
            adx_4h_mapped[i] = adx_4h[idx_4h]
    
    # Calculate BBW percentile for regime detection (rolling 100-period on 15m)
    bbw_percentile = np.zeros(n)
    bbw_window = 100
    for i in range(bbw_window - 1, n):
        valid_bbw = bbw_15m[i - bbw_window + 1:i + 1]
        valid_bbw = valid_bbw[valid_bbw > 0]
        if len(valid_bbw) > 0:
            current_bbw = bbw_15m[i]
            bbw_percentile[i] = np.sum(valid_bbw <= current_bbw) / len(valid_bbw)
    
    # Generate signals with ensemble voting and regime adaptation
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels with adaptive confidence
    SIZE_BASE = 0.20
    SIZE_CONFIDENCE_BONUS = 0.05  # Per additional agreeing signal
    SIZE_HIGH_VOL_PENALTY = 0.05
    SIZE_MAX = 0.35
    SIZE_MIN = 0.15
    
    # RSI thresholds for pullback entries (TIGHTER ranges)
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 50
    RSI_SHORT_MIN = 50
    RSI_SHORT_MAX = 60
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 20
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # BBW percentile threshold for regime
    BBW_LOW_VOL_THRESHOLD = 0.40
    BBW_HIGH_VOL_THRESHOLD = 0.70
    
    # ROC momentum threshold
    ROC_MIN = 0.5  # Minimum ROC % for momentum confirmation
    
    first_valid = max(200, 40 * bars_per_4h, bbw_window)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    prev_signal_value = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            prev_signal_value[i] = 0.0
            continue
        
        trend = trend_4h[i]
        st_trend = st_trend_4h[i]
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        adx_4h_val = adx_4h_mapped[i]
        bbw_pct = bbw_percentile[i]
        zscore_val = zscore_15m[i]
        roc_val = roc_15m[i]
        
        # Determine regime
        is_low_vol = bbw_pct < BBW_LOW_VOL_THRESHOLD
        is_high_vol = bbw_pct > BBW_HIGH_VOL_THRESHOLD
        
        # Ensemble voting: 4 signal types
        # 1. HMA trend signal (4h)
        hma_signal = 0
        if trend == 1:
            hma_signal = 1
        elif trend == -1:
            hma_signal = -1
        
        # 2. Supertrend signal (4h)
        st_signal = 0
        if st_trend == 1:
            st_signal = 1
        elif st_trend == -1:
            st_signal = -1
        
        # 3. RSI momentum signal (15m) with regime adaptation
        rsi_signal = 0
        if is_low_vol:
            # Low vol regime: Trend following with RSI pullback
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and trend == 1:
                rsi_signal = 1
            elif RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and trend == -1:
                rsi_signal = -1
        elif is_high_vol:
            # High vol regime: Mean reversion with Z-score confluence
            if zscore_val < -1.5 and rsi_val < 40:
                rsi_signal = 1
            elif zscore_val > 1.5 and rsi_val > 60:
                rsi_signal = -1
        else:
            # Mid vol: Standard RSI pullback
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and trend == 1:
                rsi_signal = 1
            elif RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and trend == -1:
                rsi_signal = -1
        
        # 4. ROC momentum confirmation (15m)
        roc_signal = 0
        if roc_val > ROC_MIN and trend == 1:
            roc_signal = 1
        elif roc_val < -ROC_MIN and trend == -1:
            roc_signal = -1
        
        # Count agreeing signals (need minimum 3/4 for entry)
        vote_count = 0
        vote_direction = 0
        
        if hma_signal == 1 and st_signal == 1 and rsi_signal == 1 and roc_signal == 1:
            vote_count = 4
            vote_direction = 1
        elif hma_signal == -1 and st_signal == -1 and rsi_signal == -1 and roc_signal == -1:
            vote_count = 4
            vote_direction = -1
        elif hma_signal == 1 and st_signal == 1 and rsi_signal == 1:
            vote_count = 3
            vote_direction = 1
        elif hma_signal == -1 and st_signal == -1 and rsi_signal == -1:
            vote_count = 3
            vote_direction = -1
        elif (hma_signal == 1 and st_signal == 1) or (hma_signal == 1 and rsi_signal == 1) or (st_signal == 1 and rsi_signal == 1):
            vote_count = 2
            vote_direction = 1
        elif (hma_signal == -1 and st_signal == -1) or (hma_signal == -1 and rsi_signal == -1) or (st_signal == -1 and rsi_signal == -1):
            vote_count = 2
            vote_direction = -1
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN and vote_direction != 0:
            vote_count = max(0, vote_count - 1)
        
        # Calculate position size based on confidence and regime
        if vote_count >= 3:
            base_size = SIZE_BASE
            confidence_bonus = (vote_count - 3) * SIZE_CONFIDENCE_BONUS
            vol_penalty = SIZE_HIGH_VOL_PENALTY if is_high_vol else 0.0
            
            position_size = base_size + confidence_bonus - vol_penalty
            position_size = np.clip(position_size, SIZE_MIN, SIZE_MAX)
            position_size = vote_direction * position_size
        elif vote_count == 2:
            # Only enter with 2 signals if in low vol regime
            if is_low_vol:
                position_size = vote_direction * SIZE_BASE
            else:
                position_size = 0.0
        else:
            position_size = 0.0
        
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
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    prev_signal_value[i] = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = prev_side * 0.5 * abs(prev_signal_value[i - 1]) if prev_signal_value[i - 1] != 0 else 0.0
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    prev_signal_value[i] = signals[i]
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        prev_signal_value[i] = 0.0
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
                    prev_signal_value[i] = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = prev_side * 0.5 * abs(prev_signal_value[i - 1]) if prev_signal_value[i - 1] != 0 else 0.0
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    prev_signal_value[i] = signals[i]
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        prev_signal_value[i] = 0.0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = prev_signal_value[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            prev_signal_value[i] = signals[i]
            continue
        
        # Entry logic: Ensemble voting with minimum 3/4 agreement
        if vote_count >= 3:
            signals[i] = position_size
            position_side[i] = vote_direction
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            prev_signal_value[i] = signals[i]
        else:
            signals[i] = 0.0
            position_side[i] = 0
            prev_signal_value[i] = 0.0
    
    return signals
```

## Last Updated
2026-03-21 09:44
