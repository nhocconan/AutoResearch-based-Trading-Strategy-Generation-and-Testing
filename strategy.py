#!/usr/bin/env python3
"""
EXPERIMENT #079 - Regime Adaptive Ensemble with MTF Voting (15m + 4h Proper HTF)
==================================================================================================
Hypothesis: Recent failures (#070, #071, #078) show ensemble strategies fail due to:
1. Improper HTF alignment (manual resampling vs mtf_data helper)
2. Too many signal changes → excessive fees
3. Position sizing not conservative enough for ensemble volatility

Key changes from #040:
- Use mtf_data helper for PROPER 4h alignment (CRITICAL - 46 strategies failed without this)
- Regime detection: BBW percentile → trend follow when low vol, reduce size when high vol
- 3-signal ensemble: HMA trend + MACD momentum + RSI mean reversion
- Confidence weighting: more signals agree = larger position (0.20 to 0.30)
- Conservative sizing: MAX 0.30 (vs 0.35 in #040) to reduce drawdown
- Fewer filters: Remove ADX/BBW minimums that killed trade count in #068
- Timeframe: 15m (proven in #031, #034, #035 with Sharpe > 7.5)

Why this should beat #040:
- Proper HTF alignment eliminates look-ahead bias from manual resampling
- Regime adaptation reduces losses in choppy high-vol periods
- Ensemble voting reduces false signals vs single-indicator strategies
- Conservative sizing (0.30 max) protects against 2022-style crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_ensemble_mtf_voting_proper_htf_15m_4h_v3"
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
    
    wma1 = pd.Series(close).rolling(window=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    ).values
    
    raw_hma = 2 * wma1 - wma2
    
    hma = pd.Series(raw_hma).rolling(window=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    ).values
    
    return np.nan_to_num(hma, nan=0.0)


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
    
    return np.nan_to_num(macd_line, nan=0.0), np.nan_to_num(signal_line, nan=0.0), np.nan_to_num(histogram, nan=0.0)


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
    
    return np.nan_to_num(upper, nan=0.0), np.nan_to_num(middle, nan=0.0), np.nan_to_num(lower, nan=0.0), np.nan_to_num(bbw, nan=0.0)


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        percentile[i] = np.sum(window <= bbw[i]) / lookback
    
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
    macd_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # Get 4h data using PROPER mtf_data helper (CRITICAL - no manual resampling!)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators for trend
        hma_4h = calculate_hma(close_4h, period=21)
        macd_4h, _, macd_hist_4h = calculate_macd(close_4h, fast=12, slow=26, signal=9)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars only)
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
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_CONSERVATIVE = 0.20
    SIZE_MODERATE = 0.25
    SIZE_FULL = 0.30  # MAX 0.30 (conservative vs 0.35 in #040)
    
    # Regime thresholds
    BBW_LOW_VOL = 0.30  # BBW percentile < 30% = low vol (trend follow)
    BBW_HIGH_VOL = 0.70  # BBW percentile > 70% = high vol (reduce size)
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # MACD histogram threshold
    MACD_MIN = 0.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5  # Slightly wider than #040's 2.0
    
    first_valid = max(200, 100, 14 * 2, 20, 26 + 9)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        macd_hist_val = macd_hist_15m[i]
        bbw_pct_val = bbw_pct_15m[i]
        
        # 4h trend signals
        hma_4h_val = hma_4h_aligned[i]
        macd_hist_4h_val = macd_hist_4h_aligned[i]
        
        # Signal 1: 4h HMA trend
        signal_1 = 0
        if hma_4h_val > 0:
            if price > hma_4h_val:
                signal_1 = 1
            elif price < hma_4h_val:
                signal_1 = -1
        
        # Signal 2: 4h MACD momentum
        signal_2 = 0
        if macd_hist_4h_val > MACD_MIN:
            signal_2 = 1
        elif macd_hist_4h_val < -MACD_MIN:
            signal_2 = -1
        
        # Signal 3: 15m RSI pullback
        signal_3 = 0
        if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
            signal_3 = 1
        elif RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
            signal_3 = -1
        
        # Ensemble voting
        vote_sum = signal_1 + signal_2 + signal_3
        
        # Regime-adaptive position sizing
        if bbw_pct_val < BBW_LOW_VOL:
            # Low volatility regime - full trend following
            size_mult = 1.0
        elif bbw_pct_val > BBW_HIGH_VOL:
            # High volatility regime - reduce size
            size_mult = 0.5
        else:
            # Normal regime
            size_mult = 0.75
        
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
            
            # Stoploss check (2.5*ATR)
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
                    signals[i] = prev_side * SIZE_FULL * size_mult * 0.5
                    position_side[i] = prev_side
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
                    signals[i] = prev_side * SIZE_FULL * size_mult * 0.5
                    position_side[i] = prev_side
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
            
            # Hold position if no exit triggered - check if ensemble still agrees
            if vote_sum * prev_side >= 1:  # At least 1 signal still agrees
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                # Ensemble disagrees - exit
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            continue
        
        # Entry logic: Ensemble voting with confidence weighting
        if vote_sum >= 2:  # At least 2 signals agree long
            base_size = SIZE_FULL if vote_sum == 3 else SIZE_MODERATE
            signals[i] = base_size * size_mult
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            
        elif vote_sum <= -2:  # At least 2 signals agree short
            base_size = SIZE_FULL if vote_sum == -3 else SIZE_MODERATE
            signals[i] = -base_size * size_mult
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals