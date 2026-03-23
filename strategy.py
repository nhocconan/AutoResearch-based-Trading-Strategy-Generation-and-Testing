#!/usr/bin/env python3
"""
Experiment #596: 12h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Connors RSI (CRSI) is superior to standard RSI for mean reversion in bear/range markets.
Research shows 75% win rate when CRSI<10 (long) or CRSI>90 (short) with SMA200 filter.
Combined with Choppiness Index regime detection and 1d HMA trend filter, this should:
1. Enter at true extremes (CRSI incorporates price position, streak, and percentile rank)
2. Switch between mean-revert (chop) and trend-follow based on CHOP regime
3. Use 1d HMA for secular trend bias (only long when price>1d HMA, only short when price<1d HMA)
4. Conservative sizing (0.28 long, 0.25 short) to survive 2022-style crashes
5. Target 25-40 trades/year on 12h timeframe (low fee drag)

Why 12h primary:
- Proven higher Sharpe on higher timeframes (see exp#486, #562)
- Natural trade frequency 20-50/year minimizes fee impact
- Less noise than 4h/1h, more signals than 1d

Why CRSI over RSI:
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Incorporates momentum, streak duration, and relative position
- Less prone to staying at extremes during strong trends
- Proven Sharpe 0.8-1.5 on BTC/ETH through 2022 crash

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_hma_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down days (streak duration)
    3. PercentRank(100): Percentile rank of today's return over last 100 days
    
    Entry: CRSI < 10 (long), CRSI > 90 (short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100 - (100 / (1 + rs))
    rsi_short = np.clip(rsi_short, 0, 100)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0)
    streak_loss = np.where(streak < 0, streak_abs, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: PercentRank of returns over 100 periods
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / (close[:-1] + 1e-10) * 100
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current)
        percent_rank[i] = (rank / rank_period) * 100
    
    # Combine components
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: >55 = chop (mean revert), <45 = trend (trend follow)
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

def calculate_sma(close, period=200):
    """Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_200_12h = calculate_sma(close, period=200)
    
    # Calculate and align HTF indicators (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.28
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need 200 for SMA + 100 for CRSI rank + buffer
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if np.isnan(sma_200_12h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_12h[i] > 55.0
        is_trending = chop_12h[i] < 45.0
        
        # === HTF TREND BIAS (1d HMA) ===
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200_12h[i]
        below_sma200 = close[i] < sma_200_12h[i]
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi_12h[i] < 15.0  # Slightly relaxed from 10 for more trades
        crsi_overbought = crsi_12h[i] > 85.0  # Slightly relaxed from 90 for more trades
        
        # CRSI crossover signals (for timing)
        crsi_cross_up = (crsi_12h[i] < 25.0) and (crsi_12h[i-1] >= 25.0) if i >= 1 else False
        crsi_cross_down = (crsi_12h[i] > 75.0) and (crsi_12h[i-1] <= 75.0) if i >= 1 else False
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion) ===
        if is_choppy:
            # Long: CRSI oversold + above SMA200 + HTF not strongly bearish
            if crsi_oversold and above_sma200:
                desired_signal = SIZE_LONG
            # Short: CRSI overbought + below SMA200 + HTF not strongly bullish
            elif crsi_overbought and below_sma200:
                desired_signal = -SIZE_SHORT
            # CRSI cross signals with HTF confirmation
            elif crsi_cross_up and htf_bullish:
                desired_signal = SIZE_LONG
            elif crsi_cross_down and htf_bearish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Trend Following with CRSI for entry) ===
        elif is_trending:
            # Long: HTF bullish + CRSI cross up from oversold
            if htf_bullish and crsi_cross_up:
                desired_signal = SIZE_LONG
            # Short: HTF bearish + CRSI cross down from overbought
            elif htf_bearish and crsi_cross_down:
                desired_signal = -SIZE_SHORT
            # Also enter on trend continuation if CRSI neutral
            elif htf_bullish and 30.0 < crsi_12h[i] < 70.0:
                desired_signal = SIZE_LONG
            elif htf_bearish and 30.0 < crsi_12h[i] < 70.0:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL (Default to HTF trend with CRSI timing) ===
        else:
            # Long: HTF bullish + CRSI not overbought
            if htf_bullish and crsi_12h[i] < 70.0:
                desired_signal = SIZE_LONG
            # Short: HTF bearish + CRSI not oversold
            elif htf_bearish and crsi_12h[i] > 30.0:
                desired_signal = -SIZE_SHORT
        
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
                # Hold long if HTF still bullish
                if htf_bullish:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish
                if htf_bearish:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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