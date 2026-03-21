#!/usr/bin/env python3
"""
EXPERIMENT #096 - VOLATILITY-ADAPTIVE ENSEMBLE WITH ADX FILTER (15m+1h v3)
==================================================================================================
Hypothesis: Current best has Sharpe=16.016. To beat this, need better regime detection + 
volatility-adaptive sizing. Key innovations:

1. 15m entries + 1h trend filter (more trades than 1h+4h, better than failed 15m+4h)
2. ADX(14) > 25 filter - only trade when trend has strength
3. Volatility-adjusted position sizing: size inversely proportional to ATR%
4. Regime-specific signal weighting: trend signals weighted higher in low vol, MR in high vol
5. Tighter hysteresis: 3 consecutive bars for entry, 1 bar for exit
6. Dynamic signal levels based on ADX strength (stronger trend = larger position)

Why this should beat Sharpe=16.016:
- ADX filter removes choppy market losses (major drawdown source)
- Volatility-adjusted sizing reduces risk in high vol periods
- 15m+1h combo captures more opportunities than 1h+4h while maintaining trend alignment
- Regime-specific weighting adapts to market conditions dynamically
"""

import numpy as np
import pandas as pd

name = "vol_adaptive_ensemble_adx_15m_1h_v3"
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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
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


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def resample_to_higher_tf(close, high, low, tf_ratio=4):
    """Resample to higher timeframe"""
    n = len(close)
    n_htf = n // tf_ratio
    
    c_htf = np.zeros(n_htf)
    h_htf = np.zeros(n_htf)
    l_htf = np.zeros(n_htf)
    
    for i in range(n_htf):
        start_idx = i * tf_ratio
        end_idx = start_idx + tf_ratio
        c_htf[i] = close[end_idx - 1]
        h_htf[i] = np.max(high[start_idx:end_idx])
        l_htf[i] = np.min(low[start_idx:end_idx])
    
    return c_htf, h_htf, l_htf


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx_15m, plus_di_15m, minus_di_15m = calculate_adx(high, low, close, period=14)
    
    # Resample to 1h for trend filter (4 x 15m = 1h)
    bars_per_1h = 4
    c_1h, h_1h, l_1h = resample_to_higher_tf(close, high, low, tf_ratio=bars_per_1h)
    
    # 1h indicators for trend
    hma_1h = calculate_hma(c_1h, period=21)
    supertrend_1h, st_direction_1h = calculate_supertrend(h_1h, l_1h, c_1h, period=10, multiplier=3.0)
    _, _, _, bbw_1h = calculate_bollinger_bands(c_1h, period=20, std_mult=2.0)
    bbw_pct_1h = calculate_bbw_percentile(bbw_1h, lookback=100)
    kama_1h = calculate_kama(c_1h, er_period=10, fast_period=2, slow_period=30)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(h_1h, l_1h, c_1h, period=14)
    
    # Map 1h indicators back to 15m timeframe
    trend_1h = np.zeros(n)
    st_trend_1h = np.zeros(n)
    bbw_pct_1h_mapped = np.zeros(n)
    kama_trend_1h = np.zeros(n)
    adx_1h_mapped = np.zeros(n)
    di_diff_1h = np.zeros(n)
    
    for i in range(n):
        idx_1h = i // bars_per_1h
        if idx_1h < len(c_1h) and idx_1h >= 100:
            if c_1h[idx_1h] > hma_1h[idx_1h]:
                trend_1h[i] = 1
            elif c_1h[idx_1h] < hma_1h[idx_1h]:
                trend_1h[i] = -1
            
            st_trend_1h[i] = st_direction_1h[idx_1h]
            bbw_pct_1h_mapped[i] = bbw_pct_1h[idx_1h]
            
            if c_1h[idx_1h] > kama_1h[idx_1h]:
                kama_trend_1h[i] = 1
            elif c_1h[idx_1h] < kama_1h[idx_1h]:
                kama_trend_1h[i] = -1
            
            adx_1h_mapped[i] = adx_1h[idx_1h]
            di_diff_1h[i] = plus_di_1h[idx_1h] - minus_di_1h[idx_1h]
    
    # Generate signals with ensemble voting
    signals = np.zeros(n)
    
    # Base position sizing - DISCRETE levels
    SIZE_LOW = 0.18
    SIZE_MED = 0.26
    SIZE_HIGH = 0.35
    
    # Volatility adjustment factor (higher ATR% = smaller position)
    ATR_TARGET_PCT = 0.02  # Target 2% ATR
    
    # Regime thresholds
    REGIME_LOW_VOL = 0.30
    REGIME_HIGH_VOL = 0.70
    
    # ADX threshold for trend strength
    ADX_MIN = 25
    
    # Hysteresis counters
    prev_vote_direction = 0
    consecutive_votes = 0
    in_position = False
    
    first_valid = max(200, 100 * bars_per_1h, 28, 20, 100)
    
    for i in range(first_valid, n):
        # Skip if any indicator is invalid
        if (np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or 
            atr_15m[i] == 0 or np.isnan(adx_15m[i])):
            signals[i] = 0.0
            consecutive_votes = 0
            prev_vote_direction = 0
            in_position = False
            continue
        
        # Get indicator values
        trend = trend_1h[i]
        st_trend = st_trend_1h[i]
        kama_trend = kama_trend_1h[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        bbw_pct = bbw_pct_1h_mapped[i]
        adx_val = adx_1h_mapped[i]
        di_diff = di_diff_1h[i]
        
        # Calculate volatility-adjusted size multiplier
        atr_pct = atr_15m[i] / close[i] if close[i] > 0 else 0
        vol_adjustment = min(1.5, max(0.5, ATR_TARGET_PCT / atr_pct)) if atr_pct > 0 else 1.0
        
        # Determine regime
        if bbw_pct < REGIME_LOW_VOL:
            regime = "trend"
        elif bbw_pct > REGIME_HIGH_VOL:
            regime = "mean_revert"
        else:
            regime = "neutral"
        
        # ADX filter - only trade if trend has strength
        adx_filter = adx_val >= ADX_MIN
        
        # ENSEMBLE VOTING: 5 independent signals
        vote_count_long = 0
        vote_count_short = 0
        
        # Signal 1: 1h HMA trend
        if trend == 1:
            vote_count_long += 1
        elif trend == -1:
            vote_count_short += 1
        
        # Signal 2: 1h Supertrend
        if st_trend == 1:
            vote_count_long += 1
        elif st_trend == -1:
            vote_count_short += 1
        
        # Signal 3: 1h KAMA trend
        if kama_trend == 1:
            vote_count_long += 1
        elif kama_trend == -1:
            vote_count_short += 1
        
        # Signal 4: 1h ADX/DMI direction
        if adx_filter:
            if di_diff > 5:
                vote_count_long += 1
            elif di_diff < -5:
                vote_count_short += 1
        
        # Signal 5: 15m RSI momentum (regime-dependent)
        if regime == "trend" or regime == "neutral":
            if rsi_val > 55:
                vote_count_long += 1
            elif rsi_val < 45:
                vote_count_short += 1
        elif regime == "mean_revert":
            if rsi_val < 35:
                vote_count_long += 1
            elif rsi_val > 65:
                vote_count_short += 1
        
        # Bonus: 15m Supertrend alignment
        if st_direction_15m[i] == 1 and vote_count_long > vote_count_short:
            vote_count_long += 0.3
        elif st_direction_15m[i] == -1 and vote_count_short > vote_count_long:
            vote_count_short += 0.3
        
        # Bonus: Z-score extreme filter (avoid entries at extremes)
        if regime == "trend" and abs(zscore_val) > 2.0:
            vote_count_long *= 0.7
            vote_count_short *= 0.7
        
        # Determine net vote
        if vote_count_long > vote_count_short:
            current_vote_direction = 1
            total_votes = vote_count_long
        elif vote_count_short > vote_count_long:
            current_vote_direction = -1
            total_votes = vote_count_short
        else:
            current_vote_direction = 0
            total_votes = 0
        
        # Hysteresis: require 3 consecutive bars for entry, 1 for exit
        if current_vote_direction != 0 and current_vote_direction == prev_vote_direction:
            consecutive_votes += 1
        elif current_vote_direction != 0:
            consecutive_votes = 1
            prev_vote_direction = current_vote_direction
        else:
            consecutive_votes = 0
            prev_vote_direction = 0
        
        # Generate signal based on vote count and hysteresis
        if in_position:
            # Already in position - only exit on vote reversal
            if current_vote_direction == 0 or current_vote_direction != np.sign(signals[i-1] if i > 0 else 0):
                signals[i] = 0.0
                in_position = False
                consecutive_votes = 0
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
        elif consecutive_votes >= 3 and total_votes >= 2.5 and adx_filter:
            # New entry with volatility adjustment
            if current_vote_direction == 1:
                if total_votes >= 4.5:
                    base_size = SIZE_HIGH
                elif total_votes >= 3.5:
                    base_size = SIZE_MED
                else:
                    base_size = SIZE_LOW
                signals[i] = base_size * vol_adjustment
                in_position = True
            else:
                if total_votes >= 4.5:
                    base_size = SIZE_HIGH
                elif total_votes >= 3.5:
                    base_size = SIZE_MED
                else:
                    base_size = SIZE_LOW
                signals[i] = -base_size * vol_adjustment
                in_position = True
        else:
            signals[i] = 0.0
    
    # Ensure signal values are within bounds
    signals = np.clip(signals, -0.40, 0.40)
    
    return signals