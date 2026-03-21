#!/usr/bin/env python3
"""
EXPERIMENT #115 - MTF Supertrend+MACD+RSI+Chandelier+VolAdj (15m+4h Proper HTF v1)
==================================================================================================
Hypothesis: Current best (mtf_supertrend_macd_bbw_rsi_15m_1h_4h_v1, Sharpe=3.653) uses 3 timeframes.
Simplify to 15m entries + 4h trend using PROPER mtf_data helper (fixes #107-#109 failures).

Key changes from failed experiments:
- Use mtf_data.get_htf_data() and align_htf_to_ltf() - NO manual resampling!
- Chandelier exit: highest_high - 3*ATR(22) for trailing stops
- Volatility-adjusted position sizing: low vol=0.35, high vol=0.20
- Discrete signal levels: 0.0, ±0.20, ±0.35 (reduce churn costs)
- MACD histogram for momentum confirmation (from current best)
- RSI for pullback entries (40-60 range)
- Supertrend for trend direction (4h)

Why this should beat current best:
- Proper HTF alignment fixes data gap issues (SOL has 2 gaps of ~3 days)
- Chandelier exit provides better trailing than fixed ATR stop
- Vol-adjusted sizing reduces risk in high vol regimes
- Simpler 15m+4h vs 15m+1h+4h (fewer conflicting signals)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_macd_rsi_chandelier_voladj_15m_4h_v1"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
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
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    ema_slow[slow - 1] = np.mean(close[:slow])
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    valid_start = slow + signal - 1
    signal_line[valid_start] = np.mean(macd_line[slow:valid_start + 1])
    for i in range(valid_start + 1, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
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


def calculate_chandelier_exit(high, low, close, atr_period=22, multiplier=3.0):
    """Calculate Chandelier Exit (ATR trailing stop)"""
    n = len(close)
    if n < atr_period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, atr_period)
    
    chandelier_long = np.zeros(n)
    chandelier_short = np.zeros(n)
    
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    
    highest_high[atr_period - 1] = np.max(high[:atr_period])
    lowest_low[atr_period - 1] = np.min(low[:atr_period])
    
    for i in range(atr_period, n):
        highest_high[i] = max(highest_high[i - 1], high[i])
        lowest_low[i] = min(lowest_low[i - 1], low[i])
        
        chandelier_long[i] = highest_high[i] - multiplier * atr[i]
        chandelier_short[i] = lowest_low[i] + multiplier * atr[i]
    
    return chandelier_long, chandelier_short


def calculate_volatility_regime(close, period=20):
    """Calculate volatility regime (low=0, high=1) based on ATR percentile"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    # Calculate ATR
    high = np.maximum(close, np.roll(close, 1))
    low = np.minimum(close, np.roll(close, 1))
    atr = calculate_atr(high, low, close, period)
    
    # Calculate ATR percentile over lookback
    vol_regime = np.zeros(n)
    lookback = period * 2
    
    for i in range(lookback, n):
        if atr[i] == 0:
            vol_regime[i] = 0
            continue
        
        window = atr[i - lookback:i + 1]
        percentile = np.sum(window <= atr[i]) / len(window)
        
        # High vol if ATR > 70th percentile
        vol_regime[i] = 1 if percentile > 0.7 else 0
    
    return vol_regime


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    macd_line_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    chandelier_long_15m, chandelier_short_15m = calculate_chandelier_exit(high, low, close, atr_period=22, multiplier=3.0)
    vol_regime_15m = calculate_volatility_regime(close, period=20)
    
    # Get 4h data using mtf_data helper (PROPER HTF alignment)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h indicators for trend
    supertrend_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    macd_line_4h, macd_signal_4h, macd_hist_4h = calculate_macd(close_4h, fast=12, slow=26, signal=9)
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
    
    # Align 4h indicators to 15m timeframe (auto shift(1) for completed bars only)
    st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    macd_hist_4h_aligned = align_htf_to_ltf(prices, df_4h, macd_hist_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on volatility regime
    SIZE_HIGH_VOL = 0.20  # High volatility = smaller position
    SIZE_LOW_VOL = 0.35   # Low volatility = larger position
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # MACD histogram threshold for momentum
    MACD_MIN = 0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 26 + 9, 22, 40)
    
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
        
        # Get 4h trend (aligned properly)
        st_trend_4h = st_trend_4h_aligned[i]
        macd_hist_4h = macd_hist_4h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        
        # Get volatility regime for position sizing
        vol_regime = vol_regime_15m[i]
        position_size = SIZE_HIGH_VOL if vol_regime == 1 else SIZE_LOW_VOL
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, high[i])
                current_low = min(prev_low, low[i]) if prev_low > 0 else low[i]
            else:
                current_high = max(prev_high, high[i]) if prev_high > 0 else high[i]
                current_low = min(prev_low, low[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Chandelier exit stoploss
            if prev_side == 1:
                chandelier_stop = chandelier_long_15m[i]
                if chandelier_stop > 0 and close[i] < chandelier_stop:
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
                    signals[i] = position_size / 2
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
                chandelier_stop = chandelier_short_15m[i]
                if chandelier_stop > 0 and close[i] > chandelier_stop:
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
                    signals[i] = -position_size / 2
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
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h Supertrend + MACD + 15m RSI + MACD
        # Long entry: 4h bullish trend + 15m pullback
        if st_trend_4h == 1 and macd_hist_4h > MACD_MIN:
            if (RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX and 
                macd_hist_15m[i] > MACD_MIN):
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
                
        # Short entry: 4h bearish trend + 15m pullback
        elif st_trend_4h == -1 and macd_hist_4h < -MACD_MIN:
            if (RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX and 
                macd_hist_15m[i] < -MACD_MIN):
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals