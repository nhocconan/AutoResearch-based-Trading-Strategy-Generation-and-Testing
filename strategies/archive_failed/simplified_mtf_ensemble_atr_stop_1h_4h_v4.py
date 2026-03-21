#!/usr/bin/env python3
"""
EXPERIMENT #097 - SIMPLIFIED MTF ENSEMBLE WITH ATR TRAILING STOP (1h+4h v4)
==================================================================================================
Hypothesis: Current best has Sharpe=16.016. Previous ensemble strategies failed due to:
1. Too many signals = too much churn/fees
2. Complex regime detection added noise
3. Poor position sizing during volatile periods

Key innovations for #097:
1. SIMPLIFIED 3-signal ensemble: HMA trend + Supertrend + RSI (fewer signals = less churn)
2. 1h entries + 4h trend filter (more stable than 15m, proven in exp#090/095)
3. ATR-based trailing stop: signal→0 when price moves 2.5*ATR against position
4. Discrete position levels: 0.0, ±0.20, ±0.35 (reduces signal changes)
5. 2-bar confirmation for entry, 1-bar for exit (balance between churn and missed entries)
6. Volatility-adjusted sizing: reduce position when ATR% is high
7. 4h ADX filter: only trade when higher timeframe has trend strength

Why this should beat Sharpe=16.016:
- Fewer signals = less fee drag (0.10% per change is significant)
- ATR trailing stop protects against large drawdowns
- 4h trend filter is more stable than 1h or 15m
- Discrete levels reduce unnecessary position adjustments
"""

import numpy as np
import pandas as pd

name = "simplified_mtf_ensemble_atr_stop_1h_4h_v4"
timeframe = "1h"
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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


def resample_to_4h(close, high, low):
    """Resample 1h data to 4h"""
    n = len(close)
    n_4h = n // 4
    
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * 4
        end_idx = start_idx + 4
        if end_idx <= n:
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = np.max(high[start_idx:end_idx])
            l_4h[i] = np.min(low[start_idx:end_idx])
    
    return c_4h, h_4h, l_4h


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    signals = np.zeros(n)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    supertrend_1h, st_direction_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # Resample to 4h for trend filter
    c_4h, h_4h, l_4h = resample_to_4h(close, high, low)
    n_4h = len(c_4h)
    
    # 4h indicators for trend
    hma_4h = calculate_hma(c_4h, period=21)
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Map 4h indicators back to 1h timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    di_diff_4h = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // 4
        if idx_4h < n_4h and idx_4h >= 30:
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            st_trend_4h[i] = st_direction_4h[idx_4h]
            adx_4h_mapped[i] = adx_4h[idx_4h]
            di_diff_4h[i] = plus_di_4h[idx_4h] - minus_di_4h[idx_4h]
    
    # Position sizing parameters
    SIZE_LOW = 0.20
    SIZE_HIGH = 0.35
    ATR_TARGET_PCT = 0.025
    ADX_MIN = 22
    
    # Tracking variables
    prev_signal = 0.0
    consecutive_votes = 0
    prev_vote_direction = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    first_valid = max(100, 30 * 4, 28, 24)
    
    for i in range(first_valid, n):
        # Skip if any indicator is invalid
        if (np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or 
            atr_1h[i] == 0 or np.isnan(adx_4h_mapped[i])):
            signals[i] = 0.0
            prev_signal = 0.0
            consecutive_votes = 0
            prev_vote_direction = 0
            entry_price = 0.0
            continue
        
        # Get indicator values
        trend = trend_4h[i]
        st_trend = st_trend_4h[i]
        adx_val = adx_4h_mapped[i]
        di_diff = di_diff_4h[i]
        rsi_val = rsi_1h[i]
        st_1h = st_direction_1h[i]
        
        # 4h ADX filter - only trade when higher timeframe has trend strength
        adx_filter = adx_val >= ADX_MIN
        
        # ENSEMBLE VOTING: 3 core signals
        vote_long = 0
        vote_short = 0
        
        # Signal 1: 4h HMA trend
        if trend == 1:
            vote_long += 1
        elif trend == -1:
            vote_short += 1
        
        # Signal 2: 4h Supertrend
        if st_trend == 1:
            vote_long += 1
        elif st_trend == -1:
            vote_short += 1
        
        # Signal 3: 1h RSI momentum (aligned with 4h trend)
        if trend == 1 and rsi_val > 50 and rsi_val < 70:
            vote_long += 1
        elif trend == -1 and rsi_val < 50 and rsi_val > 30:
            vote_short += 1
        
        # Bonus: 1h Supertrend alignment
        if st_1h == 1 and vote_long > vote_short:
            vote_long += 0.5
        elif st_1h == -1 and vote_short > vote_long:
            vote_short += 0.5
        
        # Bonus: 4h ADX/DMI confirmation
        if adx_filter:
            if di_diff > 3:
                vote_long += 0.5
            elif di_diff < -3:
                vote_short += 0.5
        
        # Determine vote direction
        if vote_long > vote_short and vote_long >= 2.0:
            current_vote = 1
            total_votes = vote_long
        elif vote_short > vote_long and vote_short >= 2.0:
            current_vote = -1
            total_votes = vote_short
        else:
            current_vote = 0
            total_votes = 0
        
        # Hysteresis: 2 consecutive bars for entry
        if current_vote != 0 and current_vote == prev_vote_direction:
            consecutive_votes += 1
        elif current_vote != 0:
            consecutive_votes = 1
            prev_vote_direction = current_vote
        else:
            consecutive_votes = 0
            prev_vote_direction = 0
        
        # Calculate volatility-adjusted size
        atr_pct = atr_1h[i] / close[i] if close[i] > 0 else 0
        vol_adjustment = min(1.3, max(0.6, ATR_TARGET_PCT / atr_pct)) if atr_pct > 0 else 1.0
        
        # Check for ATR trailing stop exit
        if prev_signal != 0.0 and entry_price > 0:
            if prev_signal > 0:  # Long position
                # Update highest close for trailing
                highest_close = max(highest_close, close[i])
                # Stop loss: 2.5 * ATR below entry or highest
                stop_long = max(entry_price - 2.5 * entry_atr, highest_close - 2.5 * atr_1h[i])
                if close[i] < stop_long:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    consecutive_votes = 0
                    continue
            else:  # Short position
                # Update lowest close for trailing
                lowest_close = min(lowest_close, close[i])
                # Stop loss: 2.5 * ATR above entry or lowest
                stop_short = min(entry_price + 2.5 * entry_atr, lowest_close + 2.5 * atr_1h[i])
                if close[i] > stop_short:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    consecutive_votes = 0
                    continue
        
        # Generate signal
        if prev_signal != 0.0:
            # Already in position
            if current_vote == 0 or current_vote != np.sign(prev_signal):
                # Exit on vote reversal or neutral
                signals[i] = 0.0
                prev_signal = 0.0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
            else:
                # Hold position
                signals[i] = prev_signal
        elif consecutive_votes >= 2 and adx_filter:
            # New entry
            if current_vote == 1:
                base_size = SIZE_HIGH if total_votes >= 3.5 else SIZE_LOW
                signals[i] = base_size * vol_adjustment
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_close = close[i]
                prev_signal = signals[i]
            else:
                base_size = SIZE_HIGH if total_votes >= 3.5 else SIZE_LOW
                signals[i] = -base_size * vol_adjustment
                entry_price = close[i]
                entry_atr = atr_1h[i]
                lowest_close = close[i]
                prev_signal = signals[i]
        else:
            signals[i] = 0.0
            prev_signal = 0.0
    
    # Clip to max position size
    signals = np.clip(signals, -0.40, 0.40)
    
    return signals