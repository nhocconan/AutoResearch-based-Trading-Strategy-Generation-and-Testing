#!/usr/bin/env python3
"""
EXPERIMENT #058 - Regime Adaptive Ensemble with MTF Voting (15m + 4h)
==================================================================================================
Hypothesis: Combine regime detection (BBW percentile) with ensemble voting from 3 signal types.
- Low volatility regime (BBW < 40th percentile): Trend-following signals get higher weight
- High volatility regime (BBW > 60th percentile): Mean-reversion signals get higher weight
- Use 15m for entries, 4h for trend filter (proven in #031, #034, #035)
- Adaptive position sizing: more signals agree = larger position (max 0.35)
- Discrete signal levels to minimize churn costs

Why this should work:
- Regime detection adapts to market conditions (trend vs mean-reversion)
- Ensemble voting reduces false signals from any single indicator
- 4h trend filter prevents counter-trend trades (proven in winners)
- Conservative sizing (0.20-0.35) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_ensemble_mtf_voting_15m_4h_v1"
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
    
    wma1 = pd.Series(close).rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    
    return np.nan_to_num(hma, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    return np.nan_to_num(rsi, nan=50.0)


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
    
    return np.nan_to_num(zscore, nan=0.0)


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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    bbw = np.zeros(n)
    for i in range(period - 1, n):
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
    
    return upper, middle, lower, bbw


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback * 100
    
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
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    _, _, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # 4h trend filter using mtf_data helper (MANDATORY)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h indicators
    hma_4h = calculate_hma(close_4h, period=21)
    _, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    
    # Align 4h indicators to 15m timeframe (auto shift for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    bbw_pct_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_pct_4h)
    
    # Generate signals with regime-adaptive ensemble voting
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_MED = 0.25
    SIZE_SMALL = 0.15
    
    # Regime thresholds
    REGIME_LOW_VOL = 40  # BBW percentile < 40 = trend regime
    REGIME_HIGH_VOL = 60  # BBW percentile > 60 = mean-reversion regime
    
    # Signal thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    ZSCORE_MAX = 2.0
    MACD_THRESHOLD = 0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 14 * 2, 20, 26 + 9)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Regime detection from 4h BBW percentile
        bbw_pct = bbw_pct_4h_aligned[i]
        
        # 4h trend filter
        trend_4h = 0
        if close[i] > hma_4h_aligned[i]:
            trend_4h = 1
        elif close[i] < hma_4h_aligned[i]:
            trend_4h = -1
        
        st_4h = st_4h_aligned[i]
        
        # Signal 1: Trend-following (HMA + Supertrend alignment)
        trend_signal = 0
        if trend_4h == 1 and st_4h == 1:
            trend_signal = 1
        elif trend_4h == -1 and st_4h == -1:
            trend_signal = -1
        
        # Signal 2: Mean-reversion (RSI extremes with 4h trend filter)
        mr_signal = 0
        if trend_4h == 1 and rsi_15m[i] < RSI_LONG_MIN:
            mr_signal = 1  # Buy pullback in uptrend
        elif trend_4h == -1 and rsi_15m[i] > (100 - RSI_SHORT_MIN):
            mr_signal = -1  # Sell rally in downtrend
        
        # Signal 3: Momentum (MACD histogram + Z-score filter)
        mom_signal = 0
        if macd_hist_15m[i] > MACD_THRESHOLD and abs(zscore_15m[i]) < ZSCORE_MAX:
            mom_signal = 1
        elif macd_hist_15m[i] < -MACD_THRESHOLD and abs(zscore_15m[i]) < ZSCORE_MAX:
            mom_signal = -1
        
        # Regime-adaptive voting
        if bbw_pct < REGIME_LOW_VOL:
            # Low volatility regime: favor trend-following
            vote_sum = trend_signal * 2 + mr_signal + mom_signal
        elif bbw_pct > REGIME_HIGH_VOL:
            # High volatility regime: favor mean-reversion
            vote_sum = trend_signal + mr_signal * 2 + mom_signal
        else:
            # Neutral regime: equal weighting
            vote_sum = trend_signal + mr_signal + mom_signal
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, close[i])
                current_low = min(prev_low, close[i]) if prev_low > 0 else close[i]
            else:
                current_high = max(prev_high, close[i]) if prev_high > 0 else close[i]
                current_low = min(prev_low, close[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] >= tp_price:
                    signals[i] = SIZE_MED
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] <= tp_price:
                    signals[i] = -SIZE_MED
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
                    if close[i] > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered - check if vote changed direction
            if (prev_side == 1 and vote_sum < 0) or (prev_side == -1 and vote_sum > 0):
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic based on vote sum and regime
        if vote_sum >= 2:
            # Strong bullish consensus
            if bbw_pct < REGIME_LOW_VOL:
                signals[i] = SIZE_FULL
            elif bbw_pct > REGIME_HIGH_VOL:
                signals[i] = SIZE_MED
            else:
                signals[i] = SIZE_MED
            
            position_side[i] = 1
            entry_price[i] = close[i]
            tp_triggered[i] = 0
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
            
        elif vote_sum <= -2:
            # Strong bearish consensus
            if bbw_pct < REGIME_LOW_VOL:
                signals[i] = -SIZE_FULL
            elif bbw_pct > REGIME_HIGH_VOL:
                signals[i] = -SIZE_MED
            else:
                signals[i] = -SIZE_MED
            
            position_side[i] = -1
            entry_price[i] = close[i]
            tp_triggered[i] = 0
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals