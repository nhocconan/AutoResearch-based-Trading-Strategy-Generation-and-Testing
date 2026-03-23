# Strategy: adaptive_kama_ensemble_multi_regime_hysteresis_15m_4h_v1

## Status
ACTIVE - Sharpe=7.760 | Return=+39607735.1% | DD=-16.6%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 6.945 | +45324.5% | -14.8% | 7962 |
| ETHUSDT | 8.145 | +763779.8% | -15.1% | 7821 |
| SOLUSDT | 8.191 | +118014100.8% | -20.0% | 8023 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 7.268 | +282.2% | -4.1% | 2443 |
| ETHUSDT | 9.184 | +1254.9% | -5.0% | 2333 |
| SOLUSDT | 10.697 | +3454.9% | -5.5% | 2221 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #056 - ADAPTIVE KAMA ENSEMBLE WITH MULTI-REGIME FILTERING
==================================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) outperforms HMA in choppy markets because it
automatically adjusts smoothing based on market efficiency. Combining KAMA with HMA gives us both
adaptive and fast trend signals. Adding multi-regime filtering (BBW + ADX + Volatility) improves
entry timing. Signal hysteresis reduces churn costs from frequent flipping.

Key innovations:
- KAMA + HMA DUAL TREND: KAMA adapts to noise, HMA provides fast trend confirmation
- MULTI-REGIME FILTER: BBW percentile + ADX + ATR volatility = 3D regime detection
- SIGNAL HYSTERESIS: Require 2-bar confirmation before entry, 1-bar for exit (reduces churn)
- CONFIDENCE SIZING: Position size scales with number of agreeing signals (0.20-0.35 range)
- 15m/4h MULTI-TF: 4h trend filter for 15m entries (proven in #049, #053, #054)
- CROSS-ASSET FILTER: Optional BTC 4h trend for ETH/SOL (disabled for BTC itself)

Why this should beat #055 (Sharpe=8.311):
- KAMA adapts better to changing volatility than fixed-period HMA
- Multi-regime filtering reduces false entries in uncertain conditions
- Hysteresis cuts transaction costs by 30-50% vs immediate signal flipping
- Based on #054's tri-TF success (Sharpe=9.485) but with adaptive KAMA

Position sizing rules (CRITICAL):
- MAX signal: 0.35 (proven to control drawdown)
- MIN signal: 0.20 (avoid tiny positions that get eaten by fees)
- Discrete levels: 0.0, 0.20, 0.28, 0.35 (reduces churn)
- Stoploss: 2.5*ATR trailing, TP at 2R then trail at 1R
"""

import numpy as np
import pandas as pd

name = "adaptive_kama_ensemble_multi_regime_hysteresis_15m_4h_v1"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts smoothing based on market efficiency ratio (ER)
    ER = |change| / sum(|individual changes|)
    High ER = trending (fast smoothing), Low ER = choppy (slow smoothing)
    """
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_percentile_rank(values, window=100):
    """Calculate rolling percentile rank"""
    n = len(values)
    percentile = np.zeros(n)
    
    for i in range(window - 1, n):
        valid_vals = values[i - window + 1:i + 1]
        valid_vals = valid_vals[~np.isnan(valid_vals)]
        if len(valid_vals) > 0:
            current_val = values[i]
            percentile[i] = np.sum(valid_vals <= current_val) / len(valid_vals)
    
    return percentile


def calculate_volatility_ratio(close, period=20):
    """Calculate volatility ratio (current ATR / average ATR)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    # Simple volatility using std of returns
    returns = np.diff(close, prepend=close[0]) / close
    vol = np.zeros(n)
    
    for i in range(period - 1, n):
        vol[i] = np.std(returns[i - period + 1:i + 1])
    
    avg_vol = np.zeros(n)
    for i in range(period * 2 - 1, n):
        avg_vol[i] = np.mean(vol[i - period + 1:i + 1])
    
    vol_ratio = np.zeros(n)
    for i in range(period * 2 - 1, n):
        if avg_vol[i] > 0:
            vol_ratio[i] = vol[i] / avg_vol[i]
    
    return vol_ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    kama_15m = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    adx_15m = calculate_adx(high, low, close, period=14)
    vol_ratio_15m = calculate_volatility_ratio(close, period=20)
    
    # Resample to 4h for trend (16 x 15m = 4h)
    bars_per_4h = 16
    n_4h = (n // bars_per_4h)
    
    # Create 4h arrays
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
    kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    kama_trend_4h = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        
        if idx_4h < n_4h and idx_4h >= 40:
            # HMA trend
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            # KAMA trend
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                kama_trend_4h[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                kama_trend_4h[i] = -1
            
            st_trend_4h[i] = st_direction_4h[idx_4h]
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
    
    # Calculate regime detection (multi-factor)
    bbw_percentile = calculate_percentile_rank(bbw_15m, window=100)
    
    # Volatility thresholds for regime
    VOL_LOW_THRESHOLD = 0.30
    VOL_HIGH_THRESHOLD = 0.70
    ADX_STRONG = 25
    ADX_WEAK = 15
    
    # Position sizing parameters (DISCRETE levels)
    SIZE_LEVELS = np.array([0.0, 0.20, 0.28, 0.35])
    
    # Signal thresholds
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    ADX_MIN = 20
    
    # Stoploss multipliers
    ATR_STOP = 2.5
    
    # Hysteresis counters (require 2 bars confirmation for entry)
    long_confirm_count = np.zeros(n, dtype=int)
    short_confirm_count = np.zeros(n, dtype=int)
    
    first_valid = max(200, 40 * bars_per_4h, 100)
    
    # Generate signals with regime-switching
    signals = np.zeros(n)
    
    # Track position state for stoploss/TP
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        kama_trend = kama_trend_4h[i]
        st_trend = st_trend_4h[i]
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        adx_4h_val = adx_4h_mapped[i]
        vol_pct = bbw_percentile[i]
        st_15m = st_direction_15m[i]
        hma_15m_val = hma_15m[i]
        kama_15m_val = kama_15m[i]
        adx_15m_val = adx_15m[i]
        vol_ratio = vol_ratio_15m[i]
        
        # Determine regime (3-factor: BBW + ADX + Vol Ratio)
        is_low_vol = vol_pct < VOL_LOW_THRESHOLD
        is_high_vol = vol_pct > VOL_HIGH_THRESHOLD
        is_strong_trend = adx_4h_val > ADX_STRONG
        is_weak_trend = adx_4h_val < ADX_WEAK
        is_vol_expanding = vol_ratio > 1.2
        is_vol_contracting = vol_ratio < 0.8
        
        # Calculate signal scores for each indicator
        # Signal 1: 4h HMA trend
        hma_signal = 0
        if trend == 1:
            hma_signal = 1
        elif trend == -1:
            hma_signal = -1
        
        # Signal 2: 4h KAMA trend (adaptive)
        kama_signal = 0
        if kama_trend == 1:
            kama_signal = 1
        elif kama_trend == -1:
            kama_signal = -1
        
        # Signal 3: 4h Supertrend
        st_signal = 0
        if st_trend == 1:
            st_signal = 1
        elif st_trend == -1:
            st_signal = -1
        
        # Signal 4: 15m RSI (pullback in trend, extreme in mean-revert)
        rsi_signal = 0
        if is_strong_trend:
            # Pullback entry in strong trend
            if trend == 1 and rsi_val <= RSI_LONG_MAX:
                rsi_signal = 1
            elif trend == -1 and rsi_val >= RSI_SHORT_MIN:
                rsi_signal = -1
        else:
            # Mean reversion in weak trend
            if rsi_val < 35:
                rsi_signal = 1
            elif rsi_val > 65:
                rsi_signal = -1
        
        # Signal 5: 15m HMA trend
        hma_15m_signal = 0
        if price > hma_15m_val:
            hma_15m_signal = 1
        elif price < hma_15m_val:
            hma_15m_signal = -1
        
        # Signal 6: 15m KAMA trend
        kama_15m_signal = 0
        if price > kama_15m_val:
            kama_15m_signal = 1
        elif price < kama_15m_val:
            kama_15m_signal = -1
        
        # Signal 7: 15m Supertrend
        st_15m_signal = 0
        if st_15m == 1:
            st_15m_signal = 1
        elif st_15m == -1:
            st_15m_signal = -1
        
        # Calculate weighted signal score based on regime
        if is_strong_trend and is_low_vol:
            # Strong trend regime: weight trend signals highest
            long_score = (
                0.25 * (hma_signal == 1) +
                0.25 * (kama_signal == 1) +
                0.20 * (st_signal == 1) +
                0.15 * (hma_15m_signal == 1) +
                0.15 * (kama_15m_signal == 1)
            )
            short_score = (
                0.25 * (hma_signal == -1) +
                0.25 * (kama_signal == -1) +
                0.20 * (st_signal == -1) +
                0.15 * (hma_15m_signal == -1) +
                0.15 * (kama_15m_signal == -1)
            )
        elif is_weak_trend and is_high_vol:
            # Mean reversion regime: weight short-term signals
            long_score = (
                0.30 * (rsi_signal == 1) +
                0.25 * (st_15m_signal == 1) +
                0.20 * (kama_15m_signal == 1) +
                0.15 * (hma_15m_signal == 1) +
                0.10 * (st_signal == 1)
            )
            short_score = (
                0.30 * (rsi_signal == -1) +
                0.25 * (st_15m_signal == -1) +
                0.20 * (kama_15m_signal == -1) +
                0.15 * (hma_15m_signal == -1) +
                0.10 * (st_signal == -1)
            )
        else:
            # Neutral regime: balanced weights
            long_score = (
                0.20 * (hma_signal == 1) +
                0.20 * (kama_signal == 1) +
                0.15 * (st_signal == 1) +
                0.15 * (rsi_signal == 1) +
                0.15 * (kama_15m_signal == 1) +
                0.15 * (st_15m_signal == 1)
            )
            short_score = (
                0.20 * (hma_signal == -1) +
                0.20 * (kama_signal == -1) +
                0.15 * (st_signal == -1) +
                0.15 * (rsi_signal == -1) +
                0.15 * (kama_15m_signal == -1) +
                0.15 * (st_15m_signal == -1)
            )
        
        # ADX filter - reduce scores if trend is very weak
        if adx_4h_val < ADX_MIN:
            long_score *= 0.8
            short_score *= 0.8
        
        # HYSTERESIS: Update confirmation counters
        if long_score >= 0.45:
            long_confirm_count[i] = long_confirm_count[i - 1] + 1 if i > 0 else 1
        else:
            long_confirm_count[i] = 0
        
        if short_score >= 0.45:
            short_confirm_count[i] = short_confirm_count[i - 1] + 1 if i > 0 else 1
        else:
            short_confirm_count[i] = 0
        
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
                stoploss_price = prev_entry - ATR_STOP * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    long_confirm_count[i] = 0
                    short_confirm_count[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP * atr
                if not prev_tp and price >= tp_price:
                    prev_signal = signals[i - 1]
                    signals[i] = prev_side * 0.5 * abs(prev_signal) if prev_signal != 0 else 0.0
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        long_confirm_count[i] = 0
                        short_confirm_count[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    long_confirm_count[i] = 0
                    short_confirm_count[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP * atr
                if not prev_tp and price <= tp_price:
                    prev_signal = signals[i - 1]
                    signals[i] = prev_side * 0.5 * abs(prev_signal) if prev_signal != 0 else 0.0
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        long_confirm_count[i] = 0
                        short_confirm_count[i] = 0
                        continue
            
            # Maintain position if signal agrees (1-bar confirmation for exit)
            if prev_side == 1:
                if long_score >= 0.40:
                    # Calculate position size based on signal agreement
                    signal_count = int(long_score * 6)  # 6 signals max
                    target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
                    target_size = max(target_size, 0.20)
                    
                    signals[i] = target_size
                    position_side[i] = 1
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
                    long_confirm_count[i] = 0
                    short_confirm_count[i] = 0
                    
            elif prev_side == -1:
                if short_score >= 0.40:
                    # Calculate position size based on signal agreement
                    signal_count = int(short_score * 6)  # 6 signals max
                    target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
                    target_size = max(target_size, 0.20)
                    
                    signals[i] = -target_size
                    position_side[i] = -1
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
                    long_confirm_count[i] = 0
                    short_confirm_count[i] = 0
            continue
        
        # Entry logic: require 2-bar confirmation (hysteresis)
        entry_threshold = 0.45
        
        if long_score >= entry_threshold and long_confirm_count[i] >= 2:
            # Calculate position size based on signal agreement
            signal_count = int(long_score * 6)  # 6 signals max
            target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
            target_size = max(target_size, 0.20)
            
            signals[i] = target_size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            short_confirm_count[i] = 0
            
        elif short_score >= entry_threshold and short_confirm_count[i] >= 2:
            # Calculate position size based on signal agreement
            signal_count = int(short_score * 6)  # 6 signals max
            target_size = SIZE_LEVELS[min(signal_count, len(SIZE_LEVELS) - 1)]
            target_size = max(target_size, 0.20)
            
            signals[i] = -target_size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            long_confirm_count[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 09:50
