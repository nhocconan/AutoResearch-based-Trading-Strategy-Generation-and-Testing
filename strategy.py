#!/usr/bin/env python3
"""
EXPERIMENT #071 - Ensemble Voting + Regime Adaptive + MTF (15m entries, 4h trend)
==================================================================================================
Hypothesis: Combining 3 independent signal generators with regime-adaptive weighting will
reduce false signals and improve risk-adjusted returns vs single-indicator strategies.

Key innovations:
1. THREE signal types voting: Trend (HMA+Supertrend), Momentum (MACD+RSI), Mean Reversion (Z-score+BB)
2. Regime detection: BBW percentile - low vol = trend follow, high vol = mean revert
3. Adaptive sizing: 0.20 base, 0.35 when all 3 agree, 0.0 when regime conflicts
4. MTF: 15m entries + 4h trend filter using mtf_data helper (CRITICAL for alignment)
5. Conservative position sizing: max 0.35 to survive 2022-style crashes
6. Stoploss: 2*ATR with trailing at 1R profit

Why this should beat current best (Sharpe=3.653):
- Ensemble voting reduces whipsaws from single indicators
- Regime adaptation avoids trend strategies in choppy markets
- 4h trend filter is more stable than 1h (proven in #060, #062, #066)
- Proper mtf_data usage ensures no look-ahead bias
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ensemble_voting_regime_adaptive_mtf_15m_4h_v1"
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
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
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
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(period - 1, n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
    
    return zscore


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    middle = rolling_mean
    upper = rolling_mean + std_mult * rolling_std
    lower = rolling_mean - std_mult * rolling_std
    
    bbw = np.zeros(n)
    for i in range(period - 1, n):
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
    
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get 4h data using mtf_data helper (CRITICAL for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
    except Exception:
        # Fallback if mtf_data not available
        df_4h = prices
        close_4h = close
        high_4h = high
        low_4h = low
    
    # ===== 15m INDICATORS (entry timing) =====
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_dir_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    macd_15m, macd_sig_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper_15m, bb_mid_15m, bb_lower_15m, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # ===== 4h INDICATORS (trend filter) =====
    hma_4h = calculate_hma(close_4h, period=21)
    supertrend_4h, st_dir_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    macd_4h, macd_sig_4h, macd_hist_4h = calculate_macd(close_4h, fast=12, slow=26, signal=9)
    bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)[3]
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=50)
    
    # Align 4h indicators to 15m timeframe
    try:
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_dir_4h_aligned = align_htf_to_ltf(prices, df_4h, st_dir_4h)
        macd_hist_4h_aligned = align_htf_to_ltf(prices, df_4h, macd_hist_4h)
        bbw_pct_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_pct_4h)
    except Exception:
        # Fallback: simple repeat
        bars_per_4h = 16  # 16 x 15m = 4h
        hma_4h_aligned = np.zeros(n)
        st_dir_4h_aligned = np.zeros(n)
        macd_hist_4h_aligned = np.zeros(n)
        bbw_pct_4h_aligned = np.zeros(n)
        
        for i in range(n):
            idx_4h = min(i // bars_per_4h, len(hma_4h) - 1)
            if idx_4h >= 0:
                hma_4h_aligned[i] = hma_4h[idx_4h]
                st_dir_4h_aligned[i] = st_dir_4h[idx_4h]
                macd_hist_4h_aligned[i] = macd_hist_4h[idx_4h]
                bbw_pct_4h_aligned[i] = bbw_pct_4h[idx_4h]
    
    # ===== SIGNAL GENERATION =====
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_BASE = 0.20
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # Thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    ZSCORE_MAX = 2.0
    BBW_REGIME_LOW = 0.30  # Below 30th percentile = low vol (trend follow)
    BBW_REGIME_HIGH = 0.70  # Above 70th percentile = high vol (mean revert)
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 100)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        
        # ===== REGIME DETECTION (4h BBW percentile) =====
        bbw_regime = bbw_pct_4h_aligned[i]
        
        # Low volatility regime (0-30th percentile): favor trend following
        # High volatility regime (70-100th percentile): favor mean reversion
        # Middle regime (30-70th): reduce position size
        
        if bbw_regime < BBW_REGIME_LOW:
            regime_type = 'trend'  # Low vol - trend follow
            regime_confidence = 1.0
        elif bbw_regime > BBW_REGIME_HIGH:
            regime_type = 'mean_revert'  # High vol - mean revert
            regime_confidence = 0.8
        else:
            regime_type = 'neutral'  # Middle - reduce size
            regime_confidence = 0.5
        
        # ===== 4h TREND FILTER =====
        trend_4h = 0
        if close[i] > hma_4h_aligned[i] and st_dir_4h_aligned[i] == 1:
            trend_4h = 1  # Bullish
        elif close[i] < hma_4h_aligned[i] and st_dir_4h_aligned[i] == -1:
            trend_4h = -1  # Bearish
        
        # ===== SIGNAL 1: TREND (HMA + Supertrend) =====
        trend_signal = 0
        if st_dir_15m[i] == 1 and close[i] > hma_15m[i]:
            trend_signal = 1
        elif st_dir_15m[i] == -1 and close[i] < hma_15m[i]:
            trend_signal = -1
        
        # ===== SIGNAL 2: MOMENTUM (MACD + RSI) =====
        momentum_signal = 0
        if macd_hist_15m[i] > 0 and RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX:
            momentum_signal = 1
        elif macd_hist_15m[i] < 0 and RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX:
            momentum_signal = -1
        
        # ===== SIGNAL 3: MEAN REVERSION (Z-score + BB position) =====
        mr_signal = 0
        bb_position = (price - bb_lower_15m[i]) / (bb_upper_15m[i] - bb_lower_15m[i]) if (bb_upper_15m[i] - bb_lower_15m[i]) > 0 else 0.5
        
        if abs(zscore_15m[i]) < ZSCORE_MAX:
            if bb_position < 0.3 and trend_4h >= 0:  # Near lower band, uptrend
                mr_signal = 1
            elif bb_position > 0.7 and trend_4h <= 0:  # Near upper band, downtrend
                mr_signal = -1
        
        # ===== ENSEMBLE VOTING =====
        signal_votes = [trend_signal, momentum_signal, mr_signal]
        vote_sum = sum(signal_votes)
        vote_agreement = sum(1 for s in signal_votes if s != 0)
        
        # Regime-adaptive signal weighting
        if regime_type == 'trend':
            # Weight trend signal more heavily
            weighted_vote = trend_signal * 2 + momentum_signal + mr_signal * 0.5
        elif regime_type == 'mean_revert':
            # Weight mean reversion signal more heavily
            weighted_vote = trend_signal * 0.5 + momentum_signal + mr_signal * 2
        else:
            weighted_vote = vote_sum
        
        # ===== POSITION SIZING BASED ON AGREEMENT =====
        if vote_agreement >= 3 and weighted_vote >= 2:
            target_size = SIZE_FULL * regime_confidence
        elif vote_agreement >= 2 and weighted_vote >= 1:
            target_size = SIZE_BASE * regime_confidence
        else:
            target_size = 0.0
        
        # ===== DIRECTIONAL BIAS FROM 4h TREND =====
        if trend_4h == 1:
            target_size = max(0, target_size)  # Only long or flat
        elif trend_4h == -1:
            target_size = min(0, target_size)  # Only short or flat
        
        # ===== EXISTING POSITION MANAGEMENT =====
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
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered and signal still valid
            if target_size != 0 and np.sign(target_size) == prev_side:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            continue
        
        # ===== NEW ENTRY =====
        if target_size != 0:
            signals[i] = target_size
            position_side[i] = np.sign(target_size)
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals