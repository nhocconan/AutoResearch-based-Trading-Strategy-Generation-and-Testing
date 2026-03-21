#!/usr/bin/env python3
"""
EXPERIMENT #078 - Ensemble Voting with Regime Detection (15m + 4h MTF)
==================================================================================================
Hypothesis: Combining 3 independent signal generators with regime-adaptive sizing will improve
Sharpe ratio while controlling drawdown. Recent ensemble attempts failed due to improper MTF
alignment and excessive position sizing.

Key innovations:
1. Proper mtf_data helper usage (get_htf_data, align_htf_to_ltf) - CRITICAL for SOL data gaps
2. 3 signal types: Trend (HMA), Momentum (RSI+MACD), Mean Reversion (Z-score+BB)
3. Regime detection: BBW percentile → trend follow in low vol, MR in high vol
4. Adaptive sizing: 0.20 (1 agree), 0.27 (2 agree), 0.35 (3 agree)
5. Max signal: 0.35 to prevent blowup (BTC -77% in 2022)
6. Stoploss: 2.5*ATR with trail at 1.5R

Why this should work:
- Proper MTF alignment prevents look-ahead bugs (46 strategies failed without this)
- Regime detection avoids wrong strategy in wrong market
- Voting reduces false signals
- Discrete sizing levels minimize churn costs
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ensemble_voting_regime_adaptive_mtf_15m_4h_v2"
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
    
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    
    wma_half = pd.Series(close).rolling(window=half, min_periods=half).apply(
        lambda x: np.sum(x * np.arange(1, half + 1)) / np.sum(np.arange(1, half + 1)), raw=True
    ).values
    
    wma_full = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1)), raw=True
    ).values
    
    wma_diff = 2 * wma_half - wma_full
    
    hma = pd.Series(wma_diff).rolling(window=sqrt_p, min_periods=sqrt_p).apply(
        lambda x: np.sum(x * np.arange(1, sqrt_p + 1)) / np.sum(np.arange(1, sqrt_p + 1)), raw=True
    ).values
    
    return np.nan_to_num(hma)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    return np.nan_to_num(rsi)


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_zscore(close, period=20):
    """Calculate Z-score"""
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = (close - rolling_mean) / np.where(rolling_std == 0, 1e-10, rolling_std)
    
    return np.nan_to_num(zscore)


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bbw = (upper - lower) / np.where(middle == 0, 1e-10, middle)
    
    return upper, middle, lower, np.nan_to_num(bbw)


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        percentile[i] = np.sum(bbw[i] >= window) / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get 4h HTF data using mtf_data helper (CRITICAL for SOL data gaps)
    df_4h = get_htf_data(prices, '4h')
    
    if df_4h is None or len(df_4h) == 0:
        return np.zeros(n)
    
    # Calculate 4h indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=48)
    rsi_4h_raw = calculate_rsi(df_4h['close'].values, period=14)
    macd_4h_raw, _, hist_4h_raw = calculate_macd(df_4h['close'].values)
    _, _, _, bbw_4h_raw = calculate_bollinger_bands(df_4h['close'].values, period=20)
    bbw_pct_4h_raw = calculate_bbw_percentile(bbw_4h_raw, lookback=100)
    
    # Align 4h indicators to 15m timeframe (auto shift(1) for completed bars)
    hma_4h = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    rsi_4h = align_htf_to_ltf(prices, df_4h, rsi_4h_raw)
    macd_hist_4h = align_htf_to_ltf(prices, df_4h, hist_4h_raw)
    bbw_4h = align_htf_to_ltf(prices, df_4h, bbw_4h_raw)
    bbw_pct_4h = align_htf_to_ltf(prices, df_4h, bbw_pct_4h_raw)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    macd_15m, _, hist_15m = calculate_macd(close)
    zscore_15m = calculate_zscore(close, period=20)
    _, middle_15m, _, bbw_15m = calculate_bollinger_bands(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    
    # Signal 1: Trend (HMA direction + MACD histogram)
    signal_trend = np.zeros(n)
    for i in range(n):
        if i < 50 or np.isnan(hma_4h[i]) or np.isnan(macd_hist_4h[i]):
            continue
        if close[i] > hma_4h[i] and macd_hist_4h[i] > 0:
            signal_trend[i] = 1
        elif close[i] < hma_4h[i] and macd_hist_4h[i] < 0:
            signal_trend[i] = -1
    
    # Signal 2: Momentum (RSI extremes + MACD cross)
    signal_momentum = np.zeros(n)
    for i in range(n):
        if i < 30 or np.isnan(rsi_15m[i]) or np.isnan(hist_15m[i]):
            continue
        if rsi_15m[i] > 45 and rsi_15m[i] < 70 and hist_15m[i] > 0:
            signal_momentum[i] = 1
        elif rsi_15m[i] < 55 and rsi_15m[i] > 30 and hist_15m[i] < 0:
            signal_momentum[i] = -1
    
    # Signal 3: Mean Reversion (Z-score + BB touch)
    signal_mr = np.zeros(n)
    for i in range(n):
        if i < 30 or np.isnan(zscore_15m[i]) or np.isnan(bbw_15m[i]):
            continue
        if zscore_15m[i] < -1.5 and close[i] < middle_15m[i]:
            signal_mr[i] = 1
        elif zscore_15m[i] > 1.5 and close[i] > middle_15m[i]:
            signal_mr[i] = -1
    
    # Regime detection: BBW percentile
    # Low vol (<30th percentile) → trend following
    # High vol (>70th percentile) → mean reversion
    regime = np.zeros(n)
    for i in range(n):
        if i < 100 or np.isnan(bbw_pct_4h[i]):
            regime[i] = 0  # neutral
        elif bbw_pct_4h[i] < 0.3:
            regime[i] = 1  # low vol - trend
        elif bbw_pct_4h[i] > 0.7:
            regime[i] = -1  # high vol - MR
        else:
            regime[i] = 0  # neutral
    
    # Voting system with regime weighting
    signals = np.zeros(n)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    SIZE_1 = 0.20  # 1 signal agrees
    SIZE_2 = 0.27  # 2 signals agree
    SIZE_3 = 0.35  # 3 signals agree
    ATR_STOP_MULT = 2.5
    ATR_TRAIL_MULT = 1.5
    
    first_valid = max(200, 100 * 4)  # Need 4h data aligned
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Weight signals by regime
        if regime[i] == 1:  # Low vol - favor trend
            vote = signal_trend[i] * 1.5 + signal_momentum[i] * 1.0 + signal_mr[i] * 0.5
        elif regime[i] == -1:  # High vol - favor MR
            vote = signal_trend[i] * 0.5 + signal_momentum[i] * 1.0 + signal_mr[i] * 1.5
        else:  # Neutral
            vote = signal_trend[i] + signal_momentum[i] + signal_mr[i]
        
        # Count agreements
        positive_votes = sum(1 for s in [signal_trend[i], signal_momentum[i], signal_mr[i]] if s > 0)
        negative_votes = sum(1 for s in [signal_trend[i], signal_momentum[i], signal_mr[i]] if s < 0)
        
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
            
            # Stoploss check
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
                
                # Take profit at 2R
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] >= tp_price:
                    signals[i] = SIZE_1 if positive_votes >= 2 else 0.0
                    position_side[i] = 1 if positive_votes >= 2 else 0
                    entry_price[i] = prev_entry if positive_votes >= 2 else 0
                    tp_triggered[i] = 1 if positive_votes >= 2 else 0
                    continue
                
                # Trail stop at 1.5R
                if prev_tp:
                    trail_stop = current_high - ATR_TRAIL_MULT * atr_15m[i]
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
                
                # Take profit at 2R
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and close[i] <= tp_price:
                    signals[i] = -SIZE_1 if negative_votes >= 2 else 0.0
                    position_side[i] = -1 if negative_votes >= 2 else 0
                    entry_price[i] = prev_entry if negative_votes >= 2 else 0
                    tp_triggered[i] = 1 if negative_votes >= 2 else 0
                    continue
                
                # Trail stop at 1.5R
                if prev_tp:
                    trail_stop = current_low + ATR_TRAIL_MULT * atr_15m[i]
                    if close[i] > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # New entry logic
        if positive_votes >= 2:
            if positive_votes == 3:
                signals[i] = SIZE_3
            elif positive_votes == 2:
                signals[i] = SIZE_2
            else:
                signals[i] = SIZE_1
            
            position_side[i] = 1
            entry_price[i] = close[i]
            tp_triggered[i] = 0
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
            
        elif negative_votes >= 2:
            if negative_votes == 3:
                signals[i] = -SIZE_3
            elif negative_votes == 2:
                signals[i] = -SIZE_2
            else:
                signals[i] = -SIZE_1
            
            position_side[i] = -1
            entry_price[i] = close[i]
            tp_triggered[i] = 0
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals