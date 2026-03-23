#!/usr/bin/env python3
"""
Experiment #790: 1h Primary + 4h/12h HTF — Regime-Adaptive Multi-Timeframe Strategy

Hypothesis: After 536 failed experiments, the key insight is:
1. 1h timeframe needs VERY STRICT filters to avoid fee drag (>100 trades/year = failure)
2. Use 12h HMA(21) for PRIMARY trend bias (slower, more reliable than 4h)
3. Use 4h Choppiness(14) for regime detection (more responsive than daily)
4. Use 1h Connors RSI for ENTRY TIMING only (pull the trigger within HTF trend)
5. Session filter (8-20 UTC) reduces trades by ~60% while keeping quality entries
6. Volume filter relaxed (0.8x avg) to ensure trades generate on BTC/ETH
7. Asymmetric sizing: 0.25 trend-follow, 0.20 mean-revert (control fees)
8. ATR(14) trailing stop at 2.5x protects from 2022-style crashes

Strategy design:
1. 12h HMA(21) = primary trend bias (aligned via mtf_data)
2. 4h Choppiness(14) = regime detection (>55 range, <45 trend)
3. 1h Connors RSI = entry timing (oversold/overbought extremes)
4. 1h Bollinger Bands(20,2.0) = mean reversion bounds
5. Session filter: only 8-20 UTC (reduces low-quality Asian session trades)
6. Volume filter: >0.8x 20-bar avg (relaxed to ensure trades)
7. Trailing stop: 2.5x ATR(14) from highest/lowest since entry
8. Discrete signals: 0.0, ±0.20, ±0.25 (minimize fee churn)

Target: Sharpe > 0.612, trades 30-80/year, ALL symbols positive Sharpe
Timeframe: 1h (with 4h/12h HTF for direction)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_hma_4h12h_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(series, period):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_rsi_streak(close, period=2):
    """Connors RSI Streak component."""
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_score = np.where(streak >= 0, streak_abs, -streak_abs)
    streak_rsi = calculate_rsi(streak_score + 100, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Connors RSI Percent Rank component."""
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10) * 100
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period:i]
        current = returns[i]
        pct_rank[i] = 100 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pct_rank_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + streak_rsi + pct_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = calculate_sma(close, period)
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    We use 55/45 for more regime switches on 4h.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.0)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF HMA for trend bias (12h is slower, more reliable)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h Choppiness for regime detection
    chop_4h_raw = calculate_choppiness(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    chop_4h_aligned = align_htf_to_ltf(prices, df_12h, chop_4h_raw)  # align to 12h structure first
    
    # Re-align 4h chop to 1h properly
    chop_4h_raw = calculate_choppiness(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    signals = np.zeros(n)
    TREND_SIZE = 0.25
    REVERT_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(bb_sma[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(chop_4h_aligned[i]):
            continue
        if np.isnan(vol_sma_1h[i]) or vol_sma_1h[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        session_active = 8 <= utc_hour <= 20
        
        # === TREND BIAS (12h HTF HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        ranging_regime = chop_4h_aligned[i] > 55
        trending_regime = chop_4h_aligned[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === VOLUME CONFIRMATION (relaxed) ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_1h[i]
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi_1h[i] < 20
        crsi_overbought = crsi_1h[i] > 80
        crsi_extreme_oversold = crsi_1h[i] < 15
        crsi_extreme_overbought = crsi_1h[i] > 85
        crsi_neutral_low = 35 < crsi_1h[i] < 50
        crsi_neutral_high = 50 < crsi_1h[i] < 65
        
        # === BOLLINGER POSITION ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        
        desired_signal = 0.0
        
        # Only trade during active session (reduces trades by ~60%)
        if session_active:
            # === RANGING REGIME LOGIC (CHOP > 55) ===
            if ranging_regime:
                # Mean reversion long: CRSI oversold + below BB lower + volume
                if crsi_oversold and below_bb_lower and volume_confirmed:
                    desired_signal = REVERT_SIZE
                
                # Mean reversion short: CRSI overbought + above BB upper + volume
                if crsi_overbought and above_bb_upper and volume_confirmed:
                    desired_signal = -REVERT_SIZE
                
                # Conservative: extreme CRSI + trend alignment
                if crsi_extreme_oversold and trend_12h_bullish:
                    desired_signal = REVERT_SIZE
                
                if crsi_extreme_overbought and trend_12h_bearish:
                    desired_signal = -REVERT_SIZE
            
            # === TRENDING REGIME LOGIC (CHOP < 45) ===
            elif trending_regime:
                # Trend pullback long: 12h bullish + CRSI neutral low + volume
                if trend_12h_bullish and crsi_neutral_low and volume_confirmed:
                    desired_signal = TREND_SIZE
                
                # Trend pullback short: 12h bearish + CRSI neutral high + volume
                if trend_12h_bearish and crsi_neutral_high and volume_confirmed:
                    desired_signal = -TREND_SIZE
                
                # Breakout continuation with volume
                if trend_12h_bullish and above_bb_upper and volume_confirmed:
                    desired_signal = TREND_SIZE
                
                if trend_12h_bearish and below_bb_lower and volume_confirmed:
                    desired_signal = -TREND_SIZE
            
            # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
            else:
                # Conservative: only extreme CRSI + trend alignment
                if crsi_extreme_oversold and trend_12h_bullish:
                    desired_signal = REVERT_SIZE
                
                if crsi_extreme_overbought and trend_12h_bearish:
                    desired_signal = -REVERT_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and CRSI not overbought
                if trend_12h_bullish and crsi_1h[i] < 75:
                    desired_signal = TREND_SIZE if trending_regime else REVERT_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if trend_12h_bearish and crsi_1h[i] > 25:
                    desired_signal = -TREND_SIZE if trending_regime else -REVERT_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or CRSI overbought
            if trend_12h_bearish and crsi_1h[i] > 70:
                desired_signal = 0.0
            # Exit if price hits BB upper in ranging regime
            if ranging_regime and above_bb_upper:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or CRSI oversold
            if trend_12h_bullish and crsi_1h[i] < 30:
                desired_signal = 0.0
            # Exit if price hits BB lower in ranging regime
            if ranging_regime and below_bb_lower:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= TREND_SIZE:
                desired_signal = TREND_SIZE
            else:
                desired_signal = REVERT_SIZE
        elif desired_signal < 0:
            if desired_signal <= -TREND_SIZE:
                desired_signal = -TREND_SIZE
            else:
                desired_signal = -REVERT_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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
        
        signals[i] = desired_signal
    
    return signals