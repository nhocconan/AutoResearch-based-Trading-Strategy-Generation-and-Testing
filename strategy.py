#!/usr/bin/env python3
"""
EXPERIMENT #086 - Regime Adaptive Ensemble with Proper MTF (15m + 4h)
==================================================================================================
Hypothesis: Recent failures (#077, #078, #079, #085) show complex ensembles whipsaw.
Winning strategies (#075, #083, #084) use simpler logic with proper MTF alignment.

Key changes from #040 (current):
- Use mtf_data helper for PROPER 4h alignment (not manual resampling which causes gaps)
- Simplify to 3-signal ensemble: HMA trend + RSI momentum + BBW regime
- Regime-adaptive: trend-follow in low BBW, mean-revert in high BBW
- Ensemble voting: 2/3 signals must agree for entry
- Position sizing: 0.35 for 3/3 agreement, 0.20 for 2/3 agreement
- Stoploss: 2.0*ATR, Take profit: 2R reduce to half, trail at 1R

Why this should beat current best (Sharpe=3.653):
- Proper MTF alignment eliminates data gap issues (SOL has 2 gaps)
- Regime detection reduces trades in choppy markets
- Ensemble voting filters false signals
- Based on winning patterns from #075, #083, #084
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_ensemble_mtf_proper_htf_15m_4h_v1"
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
    rsi[:period] = 50
    
    return rsi


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
    mask = middle > 0
    bbw[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
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
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    _, middle_15m, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # 4h indicators for trend (using PROPER mtf_data helper)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        hma_4h = calculate_hma(close_4h, period=21)
        rsi_4h = calculate_rsi(close_4h, period=14)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
    except Exception:
        # Fallback if mtf_data fails
        hma_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
        bbw_4h_aligned = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_STRONG = 0.35  # 3/3 signals agree
    SIZE_WEAK = 0.20    # 2/3 signals agree
    
    # RSI thresholds
    RSI_LONG_ENTRY = 45
    RSI_SHORT_ENTRY = 55
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # BBW regime thresholds
    BBW_LOW_REGIME = 0.30   # Below 30th percentile = low vol (trend follow)
    BBW_HIGH_REGIME = 0.70  # Above 70th percentile = high vol (mean revert)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Minimum warmup period
    first_valid = max(200, 100 + 14 * 2, 20)
    
    # Track position state
    signals = np.zeros(n)
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
        
        # === REGIME DETECTION (4h BBW percentile) ===
        bbw_4h_val = bbw_4h_aligned[i]
        bbw_4h_pct = 0.5  # Default neutral
        
        # Calculate 4h BBW percentile
        if i >= 200:
            bbw_4h_window = bbw_4h_aligned[max(0, i-100):i+1]
            bbw_4h_window = bbw_4h_window[bbw_4h_window > 0]
            if len(bbw_4h_window) > 0 and bbw_4h_val > 0:
                bbw_4h_pct = np.sum(bbw_4h_window <= bbw_4h_val) / len(bbw_4h_window)
        
        # Determine regime
        if bbw_4h_pct < BBW_LOW_REGIME:
            regime = 'trend'      # Low volatility - follow trend
        elif bbw_4h_pct > BBW_HIGH_REGIME:
            regime = 'mean_revert'  # High volatility - mean revert
        else:
            regime = 'neutral'    # Medium volatility - reduce size
        
        # === SIGNAL 1: HMA Trend (4h) ===
        hma_4h_val = hma_4h_aligned[i]
        if hma_4h_val > 0:
            if price > hma_4h_val:
                signal_hma = 1
            elif price < hma_4h_val:
                signal_hma = -1
            else:
                signal_hma = 0
        else:
            signal_hma = 0
        
        # === SIGNAL 2: RSI Momentum (15m) ===
        rsi_val = rsi_15m[i]
        if regime == 'trend':
            # In trend regime: RSI confirms direction
            if rsi_val > 50:
                signal_rsi = 1
            elif rsi_val < 50:
                signal_rsi = -1
            else:
                signal_rsi = 0
        else:
            # In mean-revert regime: RSI extremes signal reversal
            if rsi_val < RSI_OVERSOLD:
                signal_rsi = 1  # Oversold -> long
            elif rsi_val > RSI_OVERBOUGHT:
                signal_rsi = -1  # Overbought -> short
            else:
                signal_rsi = 0
        
        # === SIGNAL 3: Price vs HMA (15m) ===
        hma_15m_val = hma_15m[i]
        if hma_15m_val > 0:
            if price > hma_15m_val:
                signal_hma_15m = 1
            elif price < hma_15m_val:
                signal_hma_15m = -1
            else:
                signal_hma_15m = 0
        else:
            signal_hma_15m = 0
        
        # === ENSEMBLE VOTING ===
        votes = signal_hma + signal_rsi + signal_hma_15m
        
        # Determine target position
        if votes >= 2:
            target_signal = SIZE_STRONG if votes == 3 else SIZE_WEAK
        elif votes <= -2:
            target_signal = -SIZE_STRONG if votes == -3 else -SIZE_WEAK
        else:
            target_signal = 0.0
        
        # === MANAGE EXISTING POSITIONS ===
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
                    signals[i] = SIZE_WEAK
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
                
                # Hold or reduce based on new signal
                if target_signal <= 0:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                else:
                    signals[i] = target_signal
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    
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
                    signals[i] = -SIZE_WEAK
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
                
                # Hold or reduce based on new signal
                if target_signal >= 0:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                else:
                    signals[i] = target_signal
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
            
            continue
        
        # === NEW ENTRY LOGIC ===
        if regime == 'neutral':
            # Reduce position size in neutral regime
            target_signal = target_signal * 0.7 if target_signal != 0 else 0
        
        signals[i] = target_signal
        
        if target_signal > 0:
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        elif target_signal < 0:
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            position_side[i] = 0
            entry_price[i] = 0
    
    return signals