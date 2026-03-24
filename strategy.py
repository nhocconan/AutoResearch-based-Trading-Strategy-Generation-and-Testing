#!/usr/bin/env python3
"""
Experiment #1545: 1h Primary + 4h/1d HTF — Regime-Adaptive Pullback Strategy

Hypothesis: After 11 failed 4h experiments and multiple 1h failures, the pattern is clear:
1. Lower TF (1h/30m) strategies fail due to TOO MANY trades → fee drag kills Sharpe
2. Complex regime switching creates conflicting signals → negative Sharpe
3. Simple trend following fails in bear/range markets (2022 crash, 2025 bear)

New Approach — PROVEN pattern from research for lower TF:
- 4h HMA(21) for MACRO trend direction (not 1d - too slow for 1h entries)
- 1h Connors RSI for ENTRY timing within HTF trend (pullback entries)
- Choppiness Index(14) regime filter: CHOP>55=range(mean revert), CHOP<45=trend(follow)
- SESSION filter: only 8-20 UTC (reduces trades by ~50%, avoids Asia chop)
- VOLUME filter: volume > 0.8x 20-period avg (confirms moves)
- Position size: 0.25 (conservative for 1h volatility)
- Stoploss: 2.5x ATR trailing

Why this should work for 1h:
- HTF (4h) determines direction → fewer counter-trend trades
- Session filter drastically reduces trade count (target 30-60/year)
- Connors RSI pullback entries have 75% win rate in research
- Choppiness filter avoids trading in choppy markets (whipsaw killer)
- Discrete sizing (0.0, ±0.20, ±0.25) minimizes fee churn

Timeframe: 1h (required by experiment)
HTF: 4h HMA(21) for trend bias, 1d HMA(21) for macro confirmation
Position Size: 0.25 (smaller for 1h vs 4h)
Target: Sharpe > 0.618, DD < -30%, trades 30-60/year, >30 train, >3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_4h_hma_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - mean reversion indicator
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI(Streak): Measures consecutive up/down days
    PercentRank: Where current return ranks vs last 100 periods
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI(Streak) - consecutive up/down periods
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period:i+1]
        gain_streak = np.sum(np.where(streak_window > 0, streak_window, 0))
        loss_streak = np.sum(np.where(streak_window < 0, -streak_window, 0))
        if loss_streak > 1e-10:
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + gain_streak / loss_streak))
        else:
            streak_rsi[i] = 100.0
    
    # PercentRank - where current return ranks vs last 100
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current = returns[i]
        rank = np.sum(window < current) / len(window)
        percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate CHOP
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative for 1h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop[i] > 55.0  # Range market → mean revert
        trending_regime = chop[i] < 45.0  # Trend market → trend follow
        neutral_regime = not choppy_regime and not trending_regime
        
        # === MACRO TREND BIAS (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === MACRO CONFIRMATION (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        # Long: CRSI < 25 (oversold pullback)
        # Short: CRSI > 75 (overbought pullback)
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_neutral = 35.0 < crsi[i] < 65.0
        
        # === DESIRED SIGNAL — REGIME-ADAPTIVE ===
        desired_signal = 0.0
        long_score = 0
        short_score = 0
        
        if in_session and volume_ok:
            # LONG SETUP
            if choppy_regime:
                # Range market: mean revert long on oversold
                if crsi_oversold:
                    long_score += 3
                if hma_4h_bull:
                    long_score += 1  # HTF trend support
            elif trending_regime:
                # Trend market: follow trend on pullback
                if hma_4h_bull and hma_1d_bull:
                    long_score += 2  # Both HTF bullish
                if crsi_oversold or crsi[i] < 40.0:
                    long_score += 2  # Pullback entry
            else:
                # Neutral regime: require stronger confluence
                if hma_4h_bull and hma_1d_bull and crsi_oversold:
                    long_score += 4
            
            # SHORT SETUP
            if choppy_regime:
                # Range market: mean revert short on overbought
                if crsi_overbought:
                    short_score += 3
                if hma_4h_bear:
                    short_score += 1  # HTF trend support
            elif trending_regime:
                # Trend market: follow trend on pullback
                if hma_4h_bear and hma_1d_bear:
                    short_score += 2  # Both HTF bearish
                if crsi_overbought or crsi[i] > 60.0:
                    short_score += 2  # Pullback entry
            else:
                # Neutral regime: require stronger confluence
                if hma_4h_bear and hma_1d_bear and crsi_overbought:
                    short_score += 4
        
        # Entry thresholds — LOOSE enough to fire trades but strict enough for quality
        if long_score >= 4:
            desired_signal = BASE_SIZE
        elif short_score >= 4:
            desired_signal = -BASE_SIZE
        elif long_score >= 3 and hma_4h_bull:
            desired_signal = BASE_SIZE * 0.8
        elif short_score >= 3 and hma_4h_bear:
            desired_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals