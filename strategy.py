#!/usr/bin/env python3
"""
EXPERIMENT #074 - Confidence-Weighted Ensemble with Adaptive Regime (15m + 4h)
==================================================================================================
Hypothesis: Previous ensemble strategies failed due to too many signals causing churn and fees.
This version uses:
- Only 4 high-quality signals (HMA, Supertrend, RSI, MACD) to reduce noise
- Confidence-weighted sizing: more agreement = larger position (0.20 to 0.35)
- Dual regime detection: BBW percentile + ATR volatility for better regime classification
- 15m entries with 4h trend filter (proven from #066, #073)
- Stricter entry filters to reduce false signals and fee drain

Why this should work:
- Fewer signals = fewer changes = lower fees (0.10% per change adds up fast)
- Confidence weighting captures strong moves while staying small in uncertain markets
- Dual regime (BBW + ATR) better identifies true trend vs chop conditions
- Based on #073 success (Sharpe=0.200) but with cleaner signal logic
- MACD histogram adds momentum confirmation missing from pure trend indicators
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "confidence_weighted_ensemble_regime_15m_4h_v1"
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
    
    hma = pd.Series(2 * wma1 - wma2).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


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
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        if len(window) == lookback:
            rank = np.sum(window < bbw[i])
            percentile[i] = rank / lookback
    
    return percentile


def calculate_atr_percentile(atr, lookback=100):
    """Calculate ATR percentile for volatility regime"""
    n = len(atr)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = atr[i - lookback + 1:i + 1]
        if len(window) == lookback:
            rank = np.sum(window < atr[i])
            percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 15m indicators for entry timing ==========
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    macd_line_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    atr_pct_15m = calculate_atr_percentile(atr_15m, lookback=100)
    
    # ========== 4h indicators for trend (using mtf_data helper) ==========
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    hma_4h = calculate_hma(close_4h, period=21)
    st_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    macd_line_4h, macd_signal_4h, macd_hist_4h = calculate_macd(close_4h, fast=12, slow=26, signal=9)
    _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    atr_pct_4h = calculate_atr_percentile(calculate_atr(high_4h, low_4h, close_4h, 14), lookback=100)
    
    # Align 4h indicators to 15m timeframe (auto shift for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    st_dir_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    macd_hist_4h_aligned = align_htf_to_ltf(prices, df_4h, macd_hist_4h)
    bbw_pct_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_pct_4h)
    atr_pct_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_pct_4h)
    
    # ========== Signal generation ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on confidence (CRITICAL for drawdown control)
    SIZE_LOW = 0.20    # 1-2 signals agree
    SIZE_MED = 0.28    # 3 signals agree
    SIZE_HIGH = 0.35   # 4 signals agree (max)
    SIZE_HALF = 0.175  # Take profit reduction
    
    # Regime thresholds (dual: BBW + ATR)
    BBW_TREND_THRESHOLD = 0.40  # Below = low vol trend regime
    ATR_TREND_THRESHOLD = 0.40  # Below = low vol trend regime
    
    # Entry thresholds
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 55
    MACD_HIST_MIN = 0.0  # Must be positive for long
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(250, 150)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(macd_hist_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(st_dir_4h_aligned[i]) or np.isnan(macd_hist_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        macd_hist_val = macd_hist_15m[i]
        macd_hist_4h_val = macd_hist_4h_aligned[i]
        bbw_regime = bbw_pct_4h_aligned[i]
        atr_regime = atr_pct_4h_aligned[i]
        
        # Determine regime (dual filter: both BBW and ATR must be low for trend)
        is_trend_regime = (bbw_regime < BBW_TREND_THRESHOLD) and (atr_regime < ATR_TREND_THRESHOLD)
        
        # 4h trend signals (2 signals)
        hma_trend_4h = 1 if close[i] > hma_4h_aligned[i] else (-1 if close[i] < hma_4h_aligned[i] else 0)
        st_trend_4h = st_dir_4h_aligned[i]
        macd_trend_4h = 1 if macd_hist_4h_val > 0 else (-1 if macd_hist_4h_val < 0 else 0)
        
        # 15m momentum signals (2 signals)
        st_trend_15m = st_direction_15m[i]
        macd_trend_15m = 1 if macd_hist_val > 0 else (-1 if macd_hist_val < 0 else 0)
        
        # Signal voting with confidence scoring
        bullish_votes = sum([hma_trend_4h == 1, st_trend_4h == 1, st_trend_15m == 1, macd_trend_15m == 1])
        bearish_votes = sum([hma_trend_4h == -1, st_trend_4h == -1, st_trend_15m == -1, macd_trend_15m == -1])
        
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
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic with confidence-weighted sizing
        # Determine position size based on vote count
        if bullish_votes >= 4:
            position_size = SIZE_HIGH
        elif bullish_votes >= 3:
            position_size = SIZE_MED
        elif bullish_votes >= 2:
            position_size = SIZE_LOW
        else:
            position_size = 0.0
        
        if bearish_votes >= 4:
            position_size_short = SIZE_HIGH
        elif bearish_votes >= 3:
            position_size_short = SIZE_MED
        elif bearish_votes >= 2:
            position_size_short = SIZE_LOW
        else:
            position_size_short = 0.0
        
        # Long entry: 2+ bullish votes + RSI filter + MACD confirmation
        if bullish_votes >= 2 and hma_trend_4h == 1 and macd_trend_4h == 1:
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                macd_hist_val > MACD_HIST_MIN):
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        # Short entry: 2+ bearish votes + RSI filter + MACD confirmation
        elif bearish_votes >= 2 and hma_trend_4h == -1 and macd_trend_4h == -1:
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                macd_hist_val < -MACD_HIST_MIN):
                signals[i] = -position_size_short
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals