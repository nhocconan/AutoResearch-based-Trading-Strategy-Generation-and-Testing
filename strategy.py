#!/usr/bin/env python3
"""
Experiment #945: 1h Primary + 4h/1d HTF — Connors RSI + HMA Trend + Choppiness Regime

Hypothesis: After 674 failed strategies, the winning formula for 1h timeframe is:
1. 4h HMA(21) for trend DIRECTION (not entry timing)
2. 1d HMA(21) for macro regime filter (bull/bear bias)
3. Connors RSI (CRSI) for entry TIMING on 1h — combines RSI(3) + RSI_Streak(2) + PercentRank(100)
4. Choppiness Index(14) to distinguish range vs trend regimes
5. Session filter (8-20 UTC) + volume filter for quality entries
6. Asymmetric sizing: 0.25 for high-confidence, 0.15 for medium

Why this should work:
- CRSI has 75% win rate in academic studies (Connors Research)
- 4h/1d HTF provides trend bias without overtrading
- 1h entries give precision without fee churn of 15m/30m
- Session filter avoids low-liquidity Asian session whipsaws
- Target: 40-60 trades/year (1h with strict filters)

Key differences from failed attempts:
- CRSI instead of simple RSI (more responsive to short-term extremes)
- 4h HMA aligned properly (not manual i//16 mapping)
- Discrete signal sizes (0.0, ±0.15, ±0.25) to minimize fee churn
- Hold logic maintains position through minor pullbacks
- Stoploss via signal→0 at 2.5*ATR

Timeframe: 1h (target 40-60 trades/year with strict confluence)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_trend_4h1d_chop_session_vol_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods."""
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) — short-term momentum
    2. RSI of Up/Down streak length (2) — trend persistence
    3. Percentile rank of close over last 100 bars — relative position
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of streak length
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (absolute streak length)
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0)
    streak_loss = np.where(streak < 0, streak_abs, 0)
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_avg_gain = np.concatenate([[np.nan], streak_avg_gain])
    streak_avg_loss = np.concatenate([[np.nan], streak_avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
        streak_rsi = 100 - (100 / (1 + streak_rs))
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Component 3: Percentile rank of close over last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period - 1, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / (rank_period - 1) * 100
        percent_rank[i] = rank
    
    # Combine components
    for i in range(rank_period - 1, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

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
    """Choppiness Index — measures market choppy vs trending."""
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
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time_col):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds, convert to hours UTC
    hours = (open_time_col // (1000 * 60 * 60)) % 24
    return hours.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # === PRIMARY (1h) INDICATORS ===
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    sma_200_1h = calculate_sma(close, period=200)
    
    # Volume SMA for filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === HTF (4h) INDICATORS — aligned properly ===
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # === HTF (1d) INDICATORS — aligned properly ===
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Extract hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    HIGH_SIZE = 0.25
    MED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after 250 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(chop_1h[i]) or np.isnan(sma_200_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) — avoid Asian session whipsaws ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === VOLUME FILTER — only trade on above-average volume ===
        volume_ok = volume[i] >= 0.8 * vol_sma_20[i]
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (1h Choppiness Index) ===
        ranging_regime = chop_1h[i] > 55
        trending_regime = chop_1h[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi_1h[i] < 15
        crsi_extreme_overbought = crsi_1h[i] > 85
        crsi_oversold = crsi_1h[i] < 25
        crsi_overbought = crsi_1h[i] > 75
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200_1h[i]
        below_sma200 = close[i] < sma_200_1h[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI extreme oversold + above SMA200 + session + volume
            if crsi_extreme_oversold and above_sma200 and in_session and volume_ok:
                desired_signal = HIGH_SIZE
            # Long: CRSI oversold + 4h/1d trend support
            elif crsi_oversold and (trend_4h_bullish or macro_bull) and in_session:
                desired_signal = MED_SIZE
            
            # Short: CRSI extreme overbought + below SMA200 + session + volume
            if crsi_extreme_overbought and below_sma200 and in_session and volume_ok:
                desired_signal = -HIGH_SIZE
            # Short: CRSI overbought + 4h/1d trend support
            elif crsi_overbought and (trend_4h_bearish or macro_bear) and in_session:
                desired_signal = -MED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + CRSI pullback entry
            if trend_4h_bullish or macro_bull:
                if crsi_oversold and in_session and volume_ok:
                    desired_signal = HIGH_SIZE
                elif crsi_1h[i] < 40 and in_session:
                    desired_signal = MED_SIZE
            
            # Short: Bearish trend + CRSI rally entry
            if trend_4h_bearish or macro_bear:
                if crsi_overbought and in_session and volume_ok:
                    desired_signal = -HIGH_SIZE
                elif crsi_1h[i] > 60 and in_session:
                    desired_signal = -MED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Only trade extreme CRSI with trend confluence
            if crsi_extreme_oversold and (trend_4h_bullish or macro_bull) and in_session:
                desired_signal = MED_SIZE
            if crsi_extreme_overbought and (trend_4h_bearish or macro_bear) and in_session:
                desired_signal = -MED_SIZE
        
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
                # Hold long if 4h trend intact and CRSI not overbought
                if trend_4h_bullish and crsi_1h[i] < 75:
                    desired_signal = HIGH_SIZE
                elif macro_bull and crsi_1h[i] < 70:
                    desired_signal = MED_SIZE
            elif position_side < 0:
                # Hold short if 4h trend intact and CRSI not oversold
                if trend_4h_bearish and crsi_1h[i] > 25:
                    desired_signal = -HIGH_SIZE
                elif macro_bear and crsi_1h[i] > 30:
                    desired_signal = -MED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h + 1d trend reverses + CRSI overbought
            if trend_4h_bearish and macro_bear and crsi_1h[i] > 75:
                desired_signal = 0.0
            # Exit if CRSI reaches extreme overbought (take profit)
            if crsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h + 1d trend reverses + CRSI oversold
            if trend_4h_bullish and macro_bull and crsi_1h[i] < 25:
                desired_signal = 0.0
            # Exit if CRSI reaches extreme oversold (take profit)
            if crsi_extreme_oversold:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = HIGH_SIZE if desired_signal >= HIGH_SIZE else MED_SIZE
        elif desired_signal < 0:
            desired_signal = -HIGH_SIZE if desired_signal <= -HIGH_SIZE else -MED_SIZE
        
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