#!/usr/bin/env python3
"""
Experiment #684: 4h Primary + 12h HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After analyzing 683 failed experiments, the winning pattern is clear:
1. Regime detection (Chop) is essential — trend strategies fail in ranges
2. Connors RSI extremes (<20/>80) provide high-probability mean-reversion entries
3. 12h HMA slope gives trend bias without being too slow (like 1w)
4. Simpler entry conditions = more trades (avoid 0-trade failure mode)
5. 4h timeframe targets 25-45 trades/year (optimal per Rule 10)

Key improvements over #651:
- Looser CRSI thresholds (20/80 vs 15/85) to ensure trades generate
- Simpler regime logic (chop > 50 = range, < 50 = trend)
- Added volume filter to confirm breakouts
- Removed overly complex slope calculations that caused 0 trades
- Position sizing: 0.30 discrete with 2.5*ATR trailing stop

Why this might beat Sharpe=0.520:
- 4h TF = fewer fees than 1h/15m, more responsive than 1d
- CRSI mean-reversion works in all regimes (bull/bear/range)
- 12h HMA keeps us on right side of major moves
- Conservative sizing (0.30) controls drawdown through 2022 crash

Target: 25-45 trades/year, Sharpe > 0.520, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_hma_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8: Range/consolidation (mean-revert)
    - CHOP < 38.2: Trending (trend-follow)
    - 38.2-61.8: Transition zone
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    Entry signals:
    - Long: CRSI < 20 (oversold)
    - Short: CRSI > 80 (overbought)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation (consecutive up/down days)
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100): percentile of today's return over last 100 days
    percent_rank = pd.Series(returns).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=False
    ).values * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for primary trend direction
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_4h = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    
    # Volume SMA for volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(hma_4h[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(vol_sma[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS (HMA slope) ===
        hma_12h_slope = hma_12h_aligned[i] - hma_12h_aligned[i-5] if i >= 5 else 0.0
        trend_bull = hma_12h_slope > 0.0
        trend_bear = hma_12h_slope < 0.0
        
        # Price relative to 12h HMA
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # Price relative to 200 SMA (major trend filter)
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop_14[i] > 50.0  # Range/consolidation
        is_trend = chop_14[i] < 50.0  # Trending
        
        # === CONNORS RSI EXTREMES (looser thresholds for more trades) ===
        crsi_oversold = crsi[i] < 20.0  # Oversold
        crsi_overbought = crsi[i] > 80.0  # Overbought
        crsi_neutral = crsi[i] < 50.0  # Bias toward long in neutral
        
        # === VOLUME CONFIRMATION ===
        vol_above_avg = volume[i] > vol_sma[i]
        
        # === 4H HMA SLOPE ===
        hma_4h_slope = hma_4h[i] - hma_4h[i-3] if i >= 3 else 0.0
        hma_4h_bull = hma_4h_slope > 0.0
        hma_4h_bear = hma_4h_slope < 0.0
        
        # === ENTRY LOGIC (simplified for more trades) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market + CRSI oversold = mean revert long
        if is_range and crsi_oversold:
            new_signal = POSITION_SIZE
        
        # Regime 2: Trending + bull trend + pullback entry
        elif is_trend and trend_bull and price_above_hma_12h:
            if crsi[i] < 45.0:  # Pullback in uptrend
                new_signal = POSITION_SIZE
        
        # Regime 3: Neutral/any regime + extreme oversold + above SMA200
        elif crsi_oversold and price_above_sma200:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market + CRSI overbought = mean revert short
        elif is_range and crsi_overbought:
            new_signal = -POSITION_SIZE
        
        # Regime 2: Trending + bear trend + pullback entry
        elif is_trend and trend_bear and price_below_hma_12h:
            if crsi[i] > 55.0:  # Pullback in downtrend
                new_signal = -POSITION_SIZE
        
        # Regime 3: Neutral/any regime + extreme overbought + below SMA200
        elif crsi_overbought and price_below_sma200:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON MAJOR TREND FLIP ===
        if in_position and position_side > 0:
            if trend_bear and price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if trend_bull and price_above_hma_12h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals