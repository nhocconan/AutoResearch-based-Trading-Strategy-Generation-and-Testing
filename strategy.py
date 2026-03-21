#!/usr/bin/env python3
"""
EXPERIMENT #051 - ENSEMBLE VOTING + REGIME ADAPTIVE (15m + 1h + 4h MTF)
==================================================================================================
Hypothesis: Combining 3 independent signal types with regime-based adaptive sizing will improve
Sharpe ratio while controlling drawdown. Key innovations:
- 3 signal types: HMA trend, RSI momentum, Z-score mean reversion (vote-based)
- Regime detection: BBW percentile → trend mode in low vol, mean-revert in high vol
- Adaptive sizing: 1 signal=0.20, 2 signals=0.30, 3 signals=0.35 (confidence-based)
- MTF: 15m entries + 1h trend filter + 4h regime detection (using mtf_data helper)
- Stoploss: 2*ATR with trailing at 1R profit

Why this should beat #040 (Sharpe=5.4):
- Ensemble reduces false signals (need 2/3 agreement)
- Regime adaptation avoids trend strategies in choppy markets
- Confidence-based sizing maximizes returns when signals agree
- Based on lessons from #031, #034, #035 (15m works best)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ensemble_voting_regime_adaptive_15m_1h_4h_v1"
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


def calculate_bbw(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle = np.mean(window)
        std = np.std(window)
        upper = middle + std_mult * std
        lower = middle - std_mult * std
        
        if middle > 0:
            bbw[i] = (upper - lower) / middle
        else:
            bbw[i] = 0
    
    return bbw


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        current = bbw[i]
        percentile[i] = np.sum(window <= current) / len(window) * 100
    
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
    bbw_15m = calculate_bbw(close, period=20, std_mult=2.0)
    
    # Get 1h HTF data using mtf_data helper (MANDATORY)
    try:
        df_1h = get_htf_data(prices, '1h')
        close_1h = df_1h['close'].values
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        
        # 1h indicators
        hma_1h_raw = calculate_hma(close_1h, period=21)
        rsi_1h_raw = calculate_rsi(close_1h, period=14)
        bbw_1h_raw = calculate_bbw(close_1h, period=20, std_mult=2.0)
        
        # Align 1h indicators to 15m timeframe
        hma_1h = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
        rsi_1h = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
        bbw_1h = align_htf_to_ltf(prices, df_1h, bbw_1h_raw)
        
        # 1h trend direction
        trend_1h = np.zeros(n)
        for i in range(n):
            if close[i] > hma_1h[i] and hma_1h[i] > 0:
                trend_1h[i] = 1
            elif close[i] < hma_1h[i] and hma_1h[i] > 0:
                trend_1h[i] = -1
    except Exception:
        # Fallback if mtf_data fails
        hma_1h = np.zeros(n)
        rsi_1h = np.zeros(n)
        bbw_1h = np.zeros(n)
        trend_1h = np.zeros(n)
    
    # Get 4h HTF data for regime detection
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        
        # 4h BBW for regime
        bbw_4h_raw = calculate_bbw(close_4h, period=20, std_mult=2.0)
        bbw_4h_pct_raw = calculate_bbw_percentile(bbw_4h_raw, lookback=100)
        
        # Align to 15m
        bbw_4h_pct = align_htf_to_ltf(prices, df_4h, bbw_4h_pct_raw)
    except Exception:
        bbw_4h_pct = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on signal confidence
    SIZE_1_SIGNAL = 0.20
    SIZE_2_SIGNALS = 0.30
    SIZE_3_SIGNALS = 0.35
    SIZE_HALF = 0.175
    
    # Thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    ZSCORE_MAX = 2.0
    ATR_STOP_MULT = 2.0
    BBW_LOW_VOL = 30  # Percentile < 30 = low vol (trend mode)
    BBW_HIGH_VOL = 70  # Percentile > 70 = high vol (mean revert mode)
    
    first_valid = max(200, 14 * 2, 20, 100)
    
    # Track position state
    signals = np.zeros(n)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        bbw_regime = bbw_4h_pct[i]
        hma_1h_val = hma_1h[i]
        rsi_1h_val = rsi_1h[i]
        
        # Determine regime
        if bbw_regime < BBW_LOW_VOL:
            regime = "trend"  # Low volatility - follow trend
        elif bbw_regime > BBW_HIGH_VOL:
            regime = "mean_revert"  # High volatility - mean reversion
        else:
            regime = "neutral"  # Medium volatility - require more confirmation
        
        # Signal 1: HMA Trend (15m vs 1h alignment)
        signal_hma = 0
        if hma_1h_val > 0:
            if price > hma_15m[i] and price > hma_1h_val:
                signal_hma = 1
            elif price < hma_15m[i] and price < hma_1h_val:
                signal_hma = -1
        
        # Signal 2: RSI Momentum (15m pullback in 1h trend direction)
        signal_rsi = 0
        if trend_1h[i] == 1:  # 1h bullish
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                signal_rsi = 1
        elif trend_1h[i] == -1:  # 1h bearish
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                signal_rsi = -1
        
        # Signal 3: Z-score Mean Reversion
        signal_zscore = 0
        if regime == "mean_revert":
            if zscore_val < -ZSCORE_MAX:
                signal_zscore = 1  # Oversold - go long
            elif zscore_val > ZSCORE_MAX:
                signal_zscore = -1  # Overbought - go short
        elif regime == "trend":
            # In trend mode, only take zscore if it agrees with trend
            if trend_1h[i] == 1 and zscore_val < 0 and zscore_val > -ZSCORE_MAX:
                signal_zscore = 1
            elif trend_1h[i] == -1 and zscore_val > 0 and zscore_val < ZSCORE_MAX:
                signal_zscore = -1
        
        # Ensemble voting: count agreeing signals
        bullish_signals = sum([1 for s in [signal_hma, signal_rsi, signal_zscore] if s == 1])
        bearish_signals = sum([1 for s in [signal_hma, signal_rsi, signal_zscore] if s == -1])
        
        # Determine target signal based on vote
        target_signal = 0.0
        if bullish_signals >= 2:
            if bullish_signals == 3:
                target_signal = SIZE_3_SIGNALS
            else:
                target_signal = SIZE_2_SIGNALS
        elif bearish_signals >= 2:
            if bearish_signals == 3:
                target_signal = -SIZE_3_SIGNALS
            else:
                target_signal = -SIZE_2_SIGNALS
        elif bullish_signals == 1 or bearish_signals == 1:
            if regime == "trend":
                target_signal = SIZE_1_SIGNALS if bullish_signals == 1 else -SIZE_1_SIGNALS
        
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
                
                # Hold if signal agrees
                if target_signal > 0:
                    signals[i] = target_signal
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    
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
                
                # Hold if signal agrees
                if target_signal < 0:
                    signals[i] = target_signal
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
            continue
        
        # New entry logic
        if target_signal != 0.0:
            signals[i] = target_signal
            position_side[i] = 1 if target_signal > 0 else -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals