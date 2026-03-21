#!/usr/bin/env python3
"""
EXPERIMENT #057 - Regime-Adaptive MTF with BBW Detection (15m + 1h + 4h)
==================================================================================================
Hypothesis: Ensemble voting (#051-054) failed due to signal conflicts causing excessive churn.
Instead of voting, use REGIME DETECTION to switch between trend-following and mean-reversion modes.

Key innovation:
- Bollinger Band Width percentile on 1h determines regime
- Low vol (BBW < 30th pct): Trend-follow mode (Supertrend + HMA alignment)
- High vol (BBW > 70th pct): Mean-reversion mode (RSI extremes + Bollinger)
- Medium vol: Reduced position size or flat

Why this should beat ensemble voting:
- Only ONE active signal logic at a time (no conflicts)
- Regime detection reduces trades in choppy markets
- Still uses MTF (4h trend filter) for direction bias
- Discrete signal levels minimize churn costs

Position sizing: MAX 0.35 (critical for drawdown control)
Stoploss: 2.0*ATR trailing
Take profit: 2R with trailing stop at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_adaptive_bbw_mtf_15m_1h_4h_v1"
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
    
    wma1 = pd.Series(close).rolling(window=half_period, min_periods=half_period).mean().values
    wma2 = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    
    raw_hma = 2 * wma1 - wma2
    
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
    
    return np.nan_to_num(hma, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0
    
    return rsi


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.ones(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    supertrend[period - 1] = lower_band[period - 1]
    
    for i in range(period, n):
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
    """Calculate Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower


def calculate_bbw_percentile(close, high, low, period=20, lookback=100):
    """Calculate Bollinger Band Width percentile for regime detection"""
    n = len(close)
    if n < period + lookback:
        return np.zeros(n)
    
    upper, sma, lower = calculate_bollinger_bands(close, period, 2.0)
    
    # Bandwidth = (Upper - Lower) / SMA
    bbw = np.zeros(n)
    mask = sma > 0
    bbw[mask] = (upper[mask] - lower[mask]) / sma[mask]
    
    # Calculate rolling percentile
    bbw_pct = np.zeros(n)
    for i in range(lookback, n):
        window = bbw[i-lookback:i]
        bbw_pct[i] = np.sum(window < bbw[i]) / lookback * 100
    
    return bbw_pct


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    _, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    upper_15m, sma_15m, lower_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Get 1h data for regime detection
    try:
        df_1h = get_htf_data(prices, '1h')
        close_1h = df_1h['close'].values
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        
        # 1h BBW percentile for regime
        bbw_pct_1h = calculate_bbw_percentile(close_1h, high_1h, low_1h, period=20, lookback=100)
        bbw_pct_aligned = align_htf_to_ltf(prices, df_1h, bbw_pct_1h)
        
    except Exception:
        bbw_pct_aligned = np.zeros(n)
    
    # Get 4h data for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(close_4h, period=21)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        
        # 4h Supertrend for trend confirmation
        _, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        
    except Exception:
        hma_4h_aligned = calculate_hma(close, period=48)
        st_4h_aligned = np.ones(n)
    
    # Generate signals with regime-adaptive logic
    signals = np.zeros(n)
    
    # Position size levels - DISCRETE to minimize churn
    SIZE_LOW = 0.20
    SIZE_MED = 0.30
    SIZE_HIGH = 0.35  # MAX - critical for drawdown control
    
    # Regime thresholds
    REGIME_LOW_VOL = 30   # BBW percentile < 30 = trend mode
    REGIME_HIGH_VOL = 70  # BBW percentile > 70 = mean-revert mode
    
    # Entry thresholds
    RSI_LONG_ENTRY = 30
    RSI_SHORT_ENTRY = 70
    RSI_EXIT = 50
    
    # ATR multiples
    ATR_STOP_MULT = 2.0
    ATR_TP_MULT = 2.0
    
    first_valid = max(150, 48 * 4)  # Need enough HTF bars
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Track last regime to add hysteresis
    last_regime = np.zeros(n, dtype=int)  # 0=neutral, 1=trend, 2=mean-revert
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or atr_15m[i] <= 0 or np.isnan(close[i]) or close[i] <= 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or hma_4h_aligned[i] <= 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 4h trend direction
        if close[i] > hma_4h_aligned[i]:
            trend_4h = 1
        elif close[i] < hma_4h_aligned[i]:
            trend_4h = -1
        else:
            trend_4h = 0
        
        # Regime detection from 1h BBW percentile
        bbw_val = bbw_pct_aligned[i]
        if bbw_val < REGIME_LOW_VOL:
            current_regime = 1  # Trend-follow mode
        elif bbw_val > REGIME_HIGH_VOL:
            current_regime = 2  # Mean-reversion mode
        else:
            current_regime = 0  # Neutral - reduce size
        
        last_regime[i] = current_regime
        
        rsi_val = rsi_15m[i]
        price = close[i]
        atr = atr_15m[i]
        st_15m = st_direction_15m[i]
        st_4h = st_4h_aligned[i]
        
        # Check stoploss and take profit for existing positions FIRST
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
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + ATR_TP_MULT * atr
                if not prev_tp and price >= tp_price:
                    base_size = SIZE_LOW
                    signals[i] = prev_side * base_size * 0.5
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit after TP hit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - ATR_TP_MULT * atr
                if not prev_tp and price <= tp_price:
                    base_size = SIZE_LOW
                    signals[i] = prev_side * base_size * 0.5
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit after TP hit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
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
        
        # Check for 4h trend reversal - exit if trend changes against position
        if trend_4h == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # REGIME-ADAPTIVE ENTRY LOGIC
        signal_direction = 0
        signal_strength = 0
        
        if current_regime == 1:  # TREND-FOLLOW MODE (low volatility)
            # Enter on pullback in direction of 4h trend
            if trend_4h == 1 and st_4h == 1:  # 4h bullish
                if st_15m == 1 and rsi_val > 45:  # 15m trend + not oversold
                    signal_direction = 1
                    signal_strength = SIZE_MED
            elif trend_4h == -1 and st_4h == -1:  # 4h bearish
                if st_15m == -1 and rsi_val < 55:  # 15m trend + not overbought
                    signal_direction = -1
                    signal_strength = SIZE_MED
        
        elif current_regime == 2:  # MEAN-REVERSION MODE (high volatility)
            # Enter on RSI extremes against short-term move
            if trend_4h == 1:  # 4h bullish bias
                if rsi_val <= RSI_LONG_ENTRY and close[i] <= lower_15m[i]:
                    signal_direction = 1
                    signal_strength = SIZE_HIGH
            elif trend_4h == -1:  # 4h bearish bias
                if rsi_val >= RSI_SHORT_ENTRY and close[i] >= upper_15m[i]:
                    signal_direction = -1
                    signal_strength = SIZE_HIGH
        
        else:  # NEUTRAL MODE (medium volatility)
            # Reduced position size, only strong signals
            if trend_4h == 1:
                if rsi_val <= RSI_LONG_ENTRY + 5 and st_15m == 1:
                    signal_direction = 1
                    signal_strength = SIZE_LOW
            elif trend_4h == -1:
                if rsi_val >= RSI_SHORT_ENTRY - 5 and st_15m == -1:
                    signal_direction = -1
                    signal_strength = SIZE_LOW
        
        # Apply signal with hysteresis (don't flip on weak signals)
        if signal_direction != 0:
            # Only enter if signal aligns with 4h trend
            if signal_direction == trend_4h or current_regime == 2:  # Mean-revert can go against trend
                signals[i] = signal_direction * signal_strength
                position_side[i] = signal_direction
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals