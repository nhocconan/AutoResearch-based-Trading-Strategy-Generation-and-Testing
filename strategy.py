#!/usr/bin/env python3
"""
Experiment #660: 1h Primary + 4h/12h HTF — CRSI + Choppiness Regime + Session Filter

Hypothesis: 1h timeframe with 4h/12h HTF trend filter can capture intraday moves
while avoiding the whipsaw that kills pure 1h strategies. Key innovations:

1. Connors RSI (CRSI) - combines RSI(3) + RSI_Streak(2) + PercentRank(100)
   More responsive than standard RSI for entry timing
2. Choppiness Index regime - CHOP>55 mean revert, CHOP<45 trend follow
3. 4h HMA for immediate trend, 12h HMA for macro bias (dual HTF)
4. Session filter 8-20 UTC only (high liquidity, avoids Asian chop)
5. Volume filter >0.7x average (not too strict to ensure trades)
6. Looser CRSI thresholds (25/75 instead of 10/90) to guarantee trade frequency

Why this should beat Sharpe=0.612:
- 1h entries within 4h/12h trend = fewer false breakouts than pure 1h
- Session filter avoids low-liquidity whipsaw (major cause of 1h failures)
- CRSI more responsive than RSI for pullback entries in trends
- Dual HTF (4h+12h) provides stronger trend confirmation than single HTF
- Conservative sizing (0.25) survives crashes while allowing recovery

CRITICAL: Entry conditions LOOSER than previous 1h attempts (#650, #655, #658)
to ensure >=30 trades/train, >=3/test. Previous failures had 0 trades.

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_dualhtf_session_vol_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(streak): RSI of consecutive up/down days
    PercentRank: percentage of closes in last 100 periods that are <= current close
    
    Long signal: CRSI < 25 (oversold)
    Short signal: CRSI > 75 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(close, 3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Pad to match length
    gain = np.pad(gain, (1, 0), 'constant')
    loss = np.pad(loss, (1, 0), 'constant')
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_close = 100 - (100 / (1 + rs))
    
    # RSI of streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to absolute for RSI calculation
    streak_abs = np.abs(streak)
    streak_delta = np.diff(streak_abs)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    streak_gain = np.pad(streak_gain, (1, 0), 'constant')
    streak_loss = np.pad(streak_loss, (1, 0), 'constant')
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = np.divide(avg_streak_gain, avg_streak_loss, out=np.zeros_like(avg_streak_gain), where=avg_streak_loss != 0)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window <= close[i])
        percent_rank[i] = (count_below - 1) / (rank_period - 1) * 100
    
    # Combine into CRSI
    with np.errstate(invalid='ignore'):
        crsi = (rsi_close + rsi_streak + percent_rank) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: > 55 = chop (mean revert), < 45 = trend (trend follow)
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
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Calculate volume average (50 periods)
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative for 1h timeframe
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]):
            continue
        if np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (>0.7x average) ===
        vol_ratio = volume[i] / vol_avg[i]
        vol_ok = vol_ratio > 0.7
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_1h[i] > 55.0
        is_trending = chop_1h[i] < 45.0
        
        # === HTF TREND BIAS (4h + 12h HMA) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # Dual HTF confirmation (both agree = stronger signal)
        htf_strong_bull = htf_4h_bullish and htf_12h_bullish
        htf_strong_bear = htf_4h_bearish and htf_12h_bearish
        htf_neutral = not htf_strong_bull and not htf_strong_bear
        
        # === CRSI SIGNALS (looser thresholds for trade frequency) ===
        crsi_oversold = crsi_1h[i] < 25.0
        crsi_overbought = crsi_1h[i] > 75.0
        crsi_extreme_oversold = crsi_1h[i] < 15.0
        crsi_extreme_overbought = crsi_1h[i] > 85.0
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion) ===
        if is_choppy:
            # Long: CRSI oversold + session + volume + HTF not strongly bearish
            if crsi_oversold and in_session and vol_ok and not htf_strong_bear:
                desired_signal = SIZE
            # Short: CRSI overbought + session + volume + HTF not strongly bullish
            elif crsi_overbought and in_session and vol_ok and not htf_strong_bull:
                desired_signal = -SIZE
            # Extreme CRSI overrides HTF (strong mean reversion)
            elif crsi_extreme_oversold and in_session and vol_ok:
                desired_signal = SIZE
            elif crsi_extreme_overbought and in_session and vol_ok:
                desired_signal = -SIZE
        
        # === REGIME 2: TRENDING MARKET (Trend Follow) ===
        elif is_trending:
            # Long: HTF strong bull + CRSI not overbought + pullback entry
            if htf_strong_bull and crsi_1h[i] < 60.0 and in_session and vol_ok:
                desired_signal = SIZE
            # Short: HTF strong bear + CRSI not oversold + pullback entry
            elif htf_strong_bear and crsi_1h[i] > 40.0 and in_session and vol_ok:
                desired_signal = -SIZE
            # CRSI pullback in trend (CRSI dips then recovers)
            elif htf_strong_bull and crsi_oversold and in_session:
                desired_signal = SIZE
            elif htf_strong_bear and crsi_overbought and in_session:
                desired_signal = -SIZE
        
        # === REGIME 3: NEUTRAL/TRANSITION ===
        else:
            # Use HTF direction with CRSI filter
            if htf_strong_bull and crsi_1h[i] < 50.0 and in_session and vol_ok:
                desired_signal = SIZE
            elif htf_strong_bear and crsi_1h[i] > 50.0 and in_session and vol_ok:
                desired_signal = -SIZE
            # Extreme CRSI in neutral regime
            elif crsi_extreme_oversold and in_session and vol_ok:
                desired_signal = SIZE
            elif crsi_extreme_overbought and in_session and vol_ok:
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish OR CRSI not extremely overbought
                if (htf_4h_bullish or htf_12h_bullish) and crsi_1h[i] < 80.0:
                    desired_signal = SIZE
            elif position_side < 0:
                # Hold short if HTF still bearish OR CRSI not extremely oversold
                if (htf_4h_bearish or htf_12h_bearish) and crsi_1h[i] > 20.0:
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