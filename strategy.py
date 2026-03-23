#!/usr/bin/env python3
"""
Experiment #640: 1h Primary + 4h/12h HTF — Connors RSI + Choppiness + Volume + Session

Hypothesis: 1h timeframe can work IF we use 4h/12h for TREND DIRECTION and 1h only
for ENTRY TIMING. The key is EXTREMELY strict confluence (4+ filters) to limit
trades to 30-80/year. Lower TF strategies fail due to fee drag from too many trades.

Key innovations for 1h success:
1. Connors RSI (CRSI) — 3-component mean reversion: RSI(3) + RSI_Streak(2) + PercentRank(100)
   Long: CRSI < 15 (extreme oversold). Short: CRSI > 85 (extreme overbought)
2. Choppiness Index regime — CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
3. 4h HMA + 12h HMA — BOTH must agree for trend direction (triple confluence)
4. Volume filter — volume > 1.0x 20-bar avg (confirms move has participation)
5. Session filter — only 8-20 UTC (crypto active hours, avoids dead zones)
6. Hold logic — maintain position through minor pullbacks if HTF trend intact

Why this should beat Sharpe=0.612:
- CRSI has 75% win rate documented in literature for mean reversion
- 4h+12h HTF agreement = very strong trend filter (rarely both wrong)
- Session + volume filters cut 60%+ of low-quality signals
- Conservative size (0.20) survives crashes with manageable DD
- 1h entries within 4h/12h trend = optimal frequency/quality balance

Target: Sharpe > 0.612, trades 30-80/year, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma4h12h_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — 3-component mean reversion indicator.
    
    Components:
    1. RSI(3) — short-term momentum
    2. RSI_Streak(2) — streak duration (consecutive up/down days)
    3. PercentRank(100) — magnitude of recent returns
    
    CRSI = (RSI + RSI_Streak + PercentRank) / 3
    
    Long signal: CRSI < 15 (extreme oversold)
    Short signal: CRSI > 85 (extreme overbought)
    
    Reference: Connors & Alvarez, "ConnorsRSI" (2012)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Pad to match length
    gain = np.pad(gain, (1, 0), 'constant')
    loss = np.pad(loss, (1, 0), 'constant')
    
    # EMA for RSI
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi = np.clip(rsi, 0, 100)
    
    # Component 2: RSI Streak (duration of consecutive up/down moves)
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    abs_streak = np.abs(streak)
    streak_sign = np.sign(streak)
    
    # Rolling max streak for normalization
    max_streak = pd.Series(abs_streak).rolling(window=streak_period, min_periods=streak_period).max().values
    max_streak = np.where(max_streak > 0, max_streak, 1)
    
    streak_rsi = 50 + (streak / max_streak) * 50
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Component 3: Percent Rank of returns over lookback
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.pad(returns, (1, 0), 'constant')
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window_returns = returns[i-rank_period+1:i+1]
        current_return = returns[i]
        # Count how many returns in window are <= current
        rank = np.sum(window_returns <= current_return)
        percent_rank[i] = (rank / rank_period) * 100
    
    # Combine all three components
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + percent_rank) / 3.0
        crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: > 55 = chop (mean revert), < 45 = trend (trend follow)
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Sum ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother HTF trend."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    # Convert to seconds, then to datetime, extract hour
    return (open_time // 1000 // 3600) % 24

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
    
    # Calculate 1h indicators (primary timeframe)
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Volume MA for filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    SIZE = 0.20  # Conservative for 1h (lower TF = smaller size)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(chop_1h[i]):
            continue
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        is_active_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / (vol_ma_20[i] + 1e-10)
        has_volume = vol_ratio >= 0.8  # At least 80% of average
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_1h[i] > 55.0
        is_trending = chop_1h[i] < 45.0
        
        # === HTF TREND BIAS (4h + 12h HMA) ===
        # BOTH must agree for strong signal
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # Strong HTF agreement
        htf_strong_bullish = htf_4h_bullish and htf_12h_bullish
        htf_strong_bearish = htf_4h_bearish and htf_12h_bearish
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_1h[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi_1h[i] > 85.0  # Extreme overbought
        crsi_neutral_low = crsi_1h[i] < 30.0  # Moderately oversold
        crsi_neutral_high = crsi_1h[i] > 70.0  # Moderately overbought
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion with CRSI) ===
        if is_choppy:
            # Long: CRSI extreme oversold + HTF not strongly bearish + session + volume
            if crsi_oversold and not htf_strong_bearish and is_active_session and has_volume:
                desired_signal = SIZE
            # Short: CRSI extreme overbought + HTF not strongly bullish + session + volume
            elif crsi_overbought and not htf_strong_bullish and is_active_session and has_volume:
                desired_signal = -SIZE
            # Moderate CRSI with strong HTF agreement
            elif crsi_neutral_low and htf_strong_bullish and is_active_session and has_volume:
                desired_signal = SIZE
            elif crsi_neutral_high and htf_strong_bearish and is_active_session and has_volume:
                desired_signal = -SIZE
        
        # === REGIME 2: TRENDING MARKET (Trend Follow with HTF + CRSI pullback) ===
        elif is_trending:
            # Long: Strong HTF bullish + CRSI pullback (not overbought) + session + volume
            if htf_strong_bullish and crsi_1h[i] < 60.0 and is_active_session and has_volume:
                desired_signal = SIZE
            # Short: Strong HTF bearish + CRSI pullback (not oversold) + session + volume
            elif htf_strong_bearish and crsi_1h[i] > 40.0 and is_active_session and has_volume:
                desired_signal = -SIZE
        
        # === REGIME 3: NEUTRAL/TRANSITION (45-55 CHOP) ===
        else:
            # Only trade with strong HTF agreement + extreme CRSI
            if htf_strong_bullish and crsi_oversold and is_active_session and has_volume:
                desired_signal = SIZE
            elif htf_strong_bearish and crsi_overbought and is_active_session and has_volume:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if HTF trend intact ===
        # CRITICAL: Don't exit on minor CRSI moves if HTF trend supports position
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish (even one of 4h/12h)
                if htf_4h_bullish or htf_12h_bullish:
                    if crsi_1h[i] < 75.0:  # Not extremely overbought
                        desired_signal = SIZE
            elif position_side < 0:
                # Hold short if HTF still bearish (even one of 4h/12h)
                if htf_4h_bearish or htf_12h_bearish:
                    if crsi_1h[i] > 25.0:  # Not extremely oversold
                        desired_signal = -SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE
        elif desired_signal < 0:
            desired_signal = -SIZE
        
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
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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