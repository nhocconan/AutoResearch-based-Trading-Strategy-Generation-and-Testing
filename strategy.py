#!/usr/bin/env python3
"""
EXPERIMENT #079 - ADAPTIVE_REGIME_ENSEMBLE_15M_V3
==================================================================================================
Hypothesis: Simplify the ensemble logic while maintaining regime-aware signal weighting.
Use 15m timeframe for faster entries, 4h for trend filter. Focus on vectorized calculations
to avoid timeout issues from #078.

Key improvements over #078:
- Vectorized indicator calculations (no nested loops = no timeout)
- Cleaner regime detection using BBW + ADX
- Discrete signal levels (0.0, ±0.25, ±0.35) to reduce churn costs
- Proper position management with 2*ATR stoploss + 2R take-profit
- Signal confidence scaling for adaptive position sizing

Why 15m:
- More trade opportunities than 1h (meets ≥10 trades requirement)
- Less noise than 5m (better signal quality)
- Proven in #070 (Sharpe=1.256) and #072 (Sharpe=0.589)

Position sizing:
- MAX signal: 0.35 (never 1.0 - controls drawdown)
- Discrete levels: 0.0, ±0.25, ±0.35
- Stoploss: 2*ATR, Take-profit: 2R then trail at 1R
"""

import numpy as np
import pandas as pd

name = "adaptive_regime_ensemble_15m_v3"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing - vectorized"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = 0
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, short_period=16, long_period=48):
    """Calculate Hull Moving Average - vectorized"""
    n = len(close)
    if n < long_period:
        return np.zeros(n)
    
    def wma(x, period):
        weights = np.arange(1, period + 1)
        result = np.zeros(len(x))
        for i in range(period - 1, len(x)):
            result[i] = np.sum(x[i - period + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_short = wma(close, short_period)
    wma_long = wma(close, long_period)
    
    sqrt_long = int(np.sqrt(long_period))
    diff = 2 * wma_short - wma_long
    
    hma = wma(diff, sqrt_long)
    return hma


def calculate_zscore(close, period=20):
    """Calculate Z-score - vectorized"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = rolling_std > 0
    zscore[mask] = (close[mask] - rolling_mean[mask]) / rolling_std[mask]
    
    return zscore


def calculate_rsi(close, period=14):
    """Calculate RSI - vectorized"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isnan(rsi)] = 50
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX - vectorized"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    plus_dm[1:] = np.maximum(0, high[1:] - high[:-1])
    minus_dm[1:] = np.maximum(0, low[:-1] - low[1:])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = 0
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = atr > 0
    plus_di[mask] = 100 * plus_dm[mask] / atr[mask]
    minus_di[mask] = 100 * minus_dm[mask] / atr[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask = di_sum > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / di_sum[mask]
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    adx[np.isnan(adx)] = 0
    
    return adx


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands - vectorized"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    middle = rolling_mean
    upper = middle + std_mult * rolling_std
    lower = middle - std_mult * rolling_std
    
    bbw = np.zeros(n)
    mask = middle > 0
    bbw[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
    bbw[np.isnan(bbw)] = 0
    
    return upper, middle, lower, bbw


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend - vectorized"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    mid = (high + low) / 2
    upper_band = mid + multiplier * atr
    lower_band = mid - multiplier * atr
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i-1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
    
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
    
    # Position sizing constants (MAX 0.35 to control drawdown)
    SIZE_BASE = 0.25
    SIZE_HIGH = 0.35
    
    # Regime thresholds
    ADX_TREND_THRESHOLD = 25
    BBW_TREND_PERCENTILE = 0.40
    BBW_MR_PERCENTILE = 0.70
    
    # Z-score thresholds
    ZSCORE_EXTREME = 2.0
    ZSCORE_MODERATE = 1.0
    
    # RSI thresholds by regime
    RSI_TREND_LONG = 55
    RSI_TREND_SHORT = 45
    RSI_MR_LONG = 35
    RSI_MR_SHORT = 65
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Timeframe conversion: 4h = 16 x 15m
    bars_per_4h = 16
    
    # Base timeframe (15m) indicators
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, short_period=16, long_period=48)
    zscore_15m = calculate_zscore(close, period=20)
    adx_15m = calculate_adx(high, low, close, period=14)
    bb_upper_15m, bb_mid_15m, bb_lower_15m, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    supertrend_15m, st_dir_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # 4h timeframe indicators for trend filter
    c_4h, h_4h, l_4h = resample_to_timeframe(close, high, low, bars_per_4h)
    hma_4h = calculate_hma(c_4h, short_period=16, long_period=48)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    
    # Map 4h indicators to 15m
    hma_4h_mapped = map_tf_to_base(hma_4h, bars_per_4h, n)
    adx_4h_mapped = map_tf_to_base(adx_4h, bars_per_4h, n)
    bbw_pct_4h_mapped = map_tf_to_base(bbw_pct_4h, bars_per_4h, n)
    
    # Minimum warmup period
    first_valid = max(200, 100 * bars_per_4h, 48, 45)
    
    # Initialize output arrays
    signals = np.zeros(n)
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Track previous state for position management
    prev_signal = 0.0
    prev_side = 0
    prev_entry = 0.0
    prev_tp = False
    prev_high = 0.0
    prev_low = 0.0
    
    for i in range(first_valid, n):
        # Check for invalid data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_side = 0
            prev_entry = 0.0
            prev_tp = False
            prev_high = 0.0
            prev_low = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        adx_4h_val = adx_4h_mapped[i]
        bbw_pct = bbw_pct_4h_mapped[i]
        
        # === REGIME DETECTION ===
        is_trend_regime = (adx_4h_val > ADX_TREND_THRESHOLD) and (bbw_pct < BBW_TREND_PERCENTILE)
        is_mr_regime = bbw_pct > BBW_MR_PERCENTILE
        is_neutral = not is_trend_regime and not is_mr_regime
        
        # === 4H TREND FILTER ===
        kama_4h_idx = i // bars_per_4h
        if kama_4h_idx < len(hma_4h) and hma_4h[kama_4h_idx] > 0:
            trend_4h = 1 if c_4h[kama_4h_idx] > hma_4h[kama_4h_idx] else -1
        else:
            trend_4h = 0
        
        # === 15M SIGNALS ===
        hma_signal_15m = 1 if close[i] > hma_15m[i] else -1
        st_signal_15m = st_dir_15m[i]
        
        # Z-score signal (mean reversion)
        if zscore_val < -ZSCORE_EXTREME:
            zscore_signal = 1
        elif zscore_val > ZSCORE_EXTREME:
            zscore_signal = -1
        elif zscore_val < -ZSCORE_MODERATE:
            zscore_signal = 0.5
        elif zscore_val > ZSCORE_MODERATE:
            zscore_signal = -0.5
        else:
            zscore_signal = 0
        
        # RSI momentum signal
        if is_trend_regime:
            rsi_signal = 1 if rsi_val > RSI_TREND_LONG else (-1 if rsi_val < RSI_TREND_SHORT else 0)
        elif is_mr_regime:
            rsi_signal = 1 if rsi_val < RSI_MR_LONG else (-1 if rsi_val > RSI_MR_SHORT else 0)
        else:
            rsi_signal = 1 if rsi_val > 50 else (-1 if rsi_val < 50 else 0)
        
        # Bollinger position signal
        bb_signal = 1 if price < bb_lower_15m[i] else (-1 if price > bb_upper_15m[i] else 0)
        
        # === COMBINED SIGNAL LOGIC ===
        final_signal = 0
        signal_confidence = 0
        
        if is_trend_regime:
            if trend_4h == 1:
                trend_votes = 0
                if hma_signal_15m == 1:
                    trend_votes += 1
                if st_signal_15m == 1:
                    trend_votes += 1
                if rsi_signal >= 0:
                    trend_votes += 0.5
                
                if trend_votes >= 2:
                    final_signal = 1
                    signal_confidence = min(1.0, trend_votes / 3.0)
            
            elif trend_4h == -1:
                trend_votes = 0
                if hma_signal_15m == -1:
                    trend_votes += 1
                if st_signal_15m == -1:
                    trend_votes += 1
                if rsi_signal <= 0:
                    trend_votes += 0.5
                
                if trend_votes >= 2:
                    final_signal = -1
                    signal_confidence = min(1.0, trend_votes / 3.0)
        
        elif is_mr_regime:
            mr_votes_long = 0
            if zscore_signal >= 0.5:
                mr_votes_long += 1
            if bb_signal == 1:
                mr_votes_long += 1
            if rsi_signal == 1:
                mr_votes_long += 1
            
            mr_votes_short = 0
            if zscore_signal <= -0.5:
                mr_votes_short += 1
            if bb_signal == -1:
                mr_votes_short += 1
            if rsi_signal == -1:
                mr_votes_short += 1
            
            if mr_votes_long >= 2:
                final_signal = 1
                signal_confidence = min(1.0, mr_votes_long / 3.0)
            elif mr_votes_short >= 2:
                final_signal = -1
                signal_confidence = min(1.0, mr_votes_short / 3.0)
        
        else:
            neutral_votes = 0
            if hma_signal_15m == 1 and trend_4h >= 0:
                neutral_votes += 1
            if st_signal_15m == 1:
                neutral_votes += 1
            if rsi_signal == 1:
                neutral_votes += 1
            
            if neutral_votes >= 2:
                final_signal = 1
                signal_confidence = min(1.0, neutral_votes / 3.0)
            else:
                neutral_votes_neg = 0
                if hma_signal_15m == -1 and trend_4h <= 0:
                    neutral_votes_neg += 1
                if st_signal_15m == -1:
                    neutral_votes_neg += 1
                if rsi_signal == -1:
                    neutral_votes_neg += 1
                
                if neutral_votes_neg >= 2:
                    final_signal = -1
                    signal_confidence = min(1.0, neutral_votes_neg / 3.0)
        
        # === POSITION MANAGEMENT ===
        if prev_side != 0:
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price price if)HER) if2IRCIl ifERER #ER)ERS of0ERrERnAER:Rprevprevprev
0

 =
))prev =

 prev


           


)
               

                current    if =



221
 =


            else
)
 =


            else
           
           0           


           
            current           


            else
1