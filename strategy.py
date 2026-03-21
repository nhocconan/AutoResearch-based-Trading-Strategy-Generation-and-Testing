#!/usr/bin/env python3
"""
EXPERIMENT #083 - Regime Adaptive Ensemble Voting with Proper MTF (15m + 4h)
==================================================================================================
Hypothesis: Previous regime strategies (#071-#082) failed due to manual resampling and over-complexity.
This version uses mtf_data helper (mandatory) for proper 4h alignment, simpler 3-signal voting,
and regime-adaptive position sizing based on BBW percentile.

Key innovations:
1. PROPER MTF: Use get_htf_data() and align_htf_to_ltf() - NO manual resampling
2. 3-Signal Voting: Trend (HMA), Momentum (MACD), Mean Reversion (RSI+Zscore)
3. Regime Detection: BBW percentile → trend-follow in low vol, mean-revert in high vol
4. Adaptive Sizing: 0.20 for 2/3 agreement, 0.35 for 3/3 agreement
5. Discrete signal levels to minimize churn costs (0.0, ±0.20, ±0.35)

Why this should beat #040 and current best:
- Proper MTF alignment avoids data gap issues (SOLUSDT has 2 gaps)
- Simpler voting reduces overfitting vs complex filter chains
- Regime adaptation captures both trending and ranging markets
- Based on #073/#075 which showed voting works (Sharpe 0.2-0.27)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_ensemble_voting_mtf_proper_15m_4h_v1"
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
    
    raw = 2 * wma1 - wma2
    hma = pd.Series(raw).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
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
    mask = rolling_std > 0
    zscore[mask] = (close[mask] - rolling_mean[mask]) / rolling_std[mask]
    
    return zscore


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
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
    
    return upper, middle, lower, bbw


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        current = bbw[i]
        percentile[i] = np.sum(window <= current) / len(window)
    
    return percentile


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
    macd_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # PROPER MTF: Use mtf_data helper for 4h data (MANDATORY)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators for trend filter
        hma_4h = calculate_hma(close_4h, period=21)
        macd_4h, _, macd_hist_4h = calculate_macd(close_4h, fast=12, slow=26, signal=9)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        macd_hist_4h_aligned = align_htf_to_ltf(prices, df_4h, macd_hist_4h)
        bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_4h_aligned = np.zeros(n)
        macd_hist_4h_aligned = np.zeros(n)
        bbw_4h_aligned = np.zeros(n)
    
    # Generate signals with ensemble voting
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on agreement
    SIZE_2OF3 = 0.20  # 2 signals agree
    SIZE_3OF3 = 0.35  # 3 signals agree (full conviction)
    
    # Regime thresholds
    BBW_LOW_REGIME = 0.30   # Below 30th percentile = low vol (trend follow)
    BBW_HIGH_REGIME = 0.70  # Above 70th percentile = high vol (mean revert)
    
    # Signal thresholds
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    ZSCORE_MAX = 1.5
    MACD_MIN = 0.0
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 100, 40)  # Ensure all indicators are ready
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        macd_hist_val = macd_hist_15m[i]
        bbw_pct = bbw_pct_15m[i]
        
        # 4h trend filters
        hma_4h_val = hma_4h_aligned[i]
        macd_4h_val = macd_hist_4h_aligned[i]
        
        # Determine regime based on BBW percentile
        if bbw_pct < BBW_LOW_REGIME:
            regime = 'trend'  # Low volatility - follow trend
        elif bbw_pct > BBW_HIGH_REGIME:
            regime = 'mean_revert'  # High volatility - mean reversion
        else:
            regime = 'neutral'  # Middle - be conservative
        
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
                    signals[i] = SIZE_2OF3
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
                    signals[i] = -SIZE_2OF3
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
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ENSEMBLE VOTING: 3 independent signals
        trend_vote = 0  # HMA 4h trend
        momentum_vote = 0  # MACD histogram
        mr_vote = 0  # RSI + Z-score mean reversion
        
        # Signal 1: Trend (4h HMA)
        if hma_4h_val > 0 and price > hma_4h_val:
            trend_vote = 1
        elif hma_4h_val > 0 and price < hma_4h_val:
            trend_vote = -1
        
        # Signal 2: Momentum (15m MACD histogram)
        if macd_hist_val > MACD_MIN:
            momentum_vote = 1
        elif macd_hist_val < -MACD_MIN:
            momentum_vote = -1
        
        # Signal 3: Mean Reversion (RSI + Z-score)
        if rsi_val < RSI_LONG_MAX and zscore_val < -0.5:
            mr_vote = 1  # Oversold
        elif rsi_val > RSI_SHORT_MIN and zscore_val > 0.5:
            mr_vote = -1  # Overbought
        
        # Count votes
        long_votes = sum(1 for v in [trend_vote, momentum_vote, mr_vote] if v == 1)
        short_votes = sum(1 for v in [trend_vote, momentum_vote, mr_vote] if v == -1)
        
        # Regime-adaptive entry logic
        if regime == 'trend':
            # In low vol, require trend + momentum agreement
            if long_votes >= 2 and trend_vote == 1:
                size = SIZE_3OF3 if long_votes == 3 else SIZE_2OF3
                signals[i] = size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif short_votes >= 2 and trend_vote == -1:
                size = SIZE_3OF3 if short_votes == 3 else SIZE_2OF3
                signals[i] = -size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        elif regime == 'mean_revert':
            # In high vol, require MR signal + at least 1 other
            if mr_vote == 1 and (trend_vote == 1 or momentum_vote == 1):
                size = SIZE_2OF3
                signals[i] = size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif mr_vote == -1 and (trend_vote == -1 or momentum_vote == -1):
                size = SIZE_2OF3
                signals[i] = -size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            # Neutral regime - require 3/3 agreement
            if long_votes == 3:
                signals[i] = SIZE_3OF3
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif short_votes == 3:
                signals[i] = -SIZE_3OF3
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals