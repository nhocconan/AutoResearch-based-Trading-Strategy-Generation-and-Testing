#!/usr/bin/env python3
"""
EXPERIMENT #092 - Ensemble Voting + Regime Adaptive MTF (15m entries, 4h trend)
==================================================================================================
Hypothesis: Recent failures (#086, #090) show regime-based strategies can blow up if not properly
constrained. This strategy combines:
1. Ensemble voting: 3 independent signals (Supertrend, HMA, RSI pullback) - majority rules
2. Regime detection: BBW percentile - trend follow in low vol, reduce size in high vol
3. Proper MTF: Use mtf_data helper for 4h trend filter (proven in best strategies)
4. Conservative sizing: 0.20 base, scale to 0.30 when 3/3 signals agree
5. Tight stoploss: 2.0 ATR with trailing

Why this should work:
- 15m entries + 4h trend worked in current best (Sharpe=3.653)
- Ensemble voting reduces false signals (needs 2/3 agreement minimum)
- Regime-adaptive sizing prevents over-exposure in choppy markets
- mtf_data helper ensures proper 4h alignment (fixes #086, #090 alignment bugs)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ensemble_voting_regime_mtf_proper_15m_4h_v2"
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
    
    rolling = pd.Series(close).rolling(window=period, min_periods=period)
    middle = rolling.mean().values
    std = rolling.std().values
    
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
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # Get 4h data using mtf_data helper (CRITICAL - proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators for trend filter
        hma_4h = calculate_hma(close_4h, period=21)
        _, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        rsi_4h = calculate_rsi(close_4h, period=14)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_4h_aligned = np.zeros(n)
        st_4h_aligned = np.ones(n)
        rsi_4h_aligned = np.full(n, 50.0)
    
    # Generate signals with ensemble voting
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on vote count
    SIZE_1VOTE = 0.15   # 1 signal agrees (minimum)
    SIZE_2VOTE = 0.25   # 2 signals agree
    SIZE_3VOTE = 0.35   # 3 signals agree (maximum)
    SIZE_HALF = 0.175   # Take profit reduction
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Regime thresholds (BBW percentile)
    REGIME_LOW_VOL = 0.30   # Below 30th percentile = trend regime
    REGIME_HIGH_VOL = 0.70  # Above 70th percentile = mean reversion regime
    
    first_valid = max(200, 100, 40)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        bbw_pct = bbw_pct_15m[i]
        
        # === ENSEMBLE VOTING LOGIC ===
        # Signal 1: Supertrend direction (15m)
        vote_st = st_direction_15m[i]
        
        # Signal 2: HMA trend (15m)
        vote_hma = 0
        if hma_15m[i] > 0:
            vote_hma = 1 if close[i] > hma_15m[i] else -1
        
        # Signal 3: RSI pullback + 4h trend filter
        vote_rsi = 0
        hma_4h_val = hma_4h_aligned[i]
        st_4h_val = st_4h_aligned[i]
        rsi_4h_val = rsi_4h_aligned[i]
        
        # 4h trend must agree with 15m for RSI signal
        if hma_4h_val > 0 and st_4h_val > 0 and rsi_4h_val > 50:
            # Bullish 4h - look for long pullback
            if RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX:
                vote_rsi = 1
        elif hma_4h_val > 0 and st_4h_val < 0 and rsi_4h_val < 50:
            # Bearish 4h - look for short pullback
            if RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX:
                vote_rsi = -1
        
        # Count votes
        votes = [vote_st, vote_hma, vote_rsi]
        long_votes = sum(1 for v in votes if v == 1)
        short_votes = sum(1 for v in votes if v == -1)
        
        # Determine signal direction and strength
        signal_direction = 0
        vote_count = 0
        
        if long_votes >= 2 and long_votes > short_votes:
            signal_direction = 1
            vote_count = long_votes
        elif short_votes >= 2 and short_votes > long_votes:
            signal_direction = -1
            vote_count = short_votes
        
        # Regime-adaptive sizing (reduce in high vol)
        if bbw_pct > REGIME_HIGH_VOL:
            vote_count = max(1, vote_count - 1)  # Reduce vote count in high vol
        
        # Calculate position size based on vote count
        if vote_count == 1:
            target_size = SIZE_1VOTE * signal_direction
        elif vote_count == 2:
            target_size = SIZE_2VOTE * signal_direction
        elif vote_count >= 3:
            target_size = SIZE_3VOTE * signal_direction
        else:
            target_size = 0.0
        
        # === EXISTING POSITION MANAGEMENT ===
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
            
            # Hold position if no exit triggered AND signal still agrees
            if (prev_side == 1 and signal_direction >= 0) or (prev_side == -1 and signal_direction <= 0):
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                # Signal reversed - exit position
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            continue
        
        # === NEW ENTRY LOGIC ===
        if target_size != 0.0:
            signals[i] = target_size
            position_side[i] = 1 if target_size > 0 else -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals