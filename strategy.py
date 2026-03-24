#!/usr/bin/env python3
"""
Experiment #015: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + HTF Trend

Hypothesis: After 14 experiments, the pattern is clear - over-filtering kills trades.
This strategy combines proven edges with LOOSE thresholds to ensure trade generation:

1. Connors RSI (CRSI): 3-component RSI with 75% win rate in literature
   - RSI(3) + RSI_Streak(2) + PercentRank(100) / 3
   - Long when CRSI < 20, Short when CRSI > 80 (loose thresholds)

2. Choppiness Index Regime Filter:
   - CHOP > 55 = range (favor mean reversion entries)
   - CHOP < 45 = trend (favor trend-following entries)
   - This adapts to market conditions automatically

3. 4h HMA Trend Bias (HTF):
   - Only long when 4h HMA bullish
   - Only short when 4h HMA bearish
   - Prevents fighting the higher timeframe trend

4. Session Filter (8-20 UTC):
   - Only trade during high liquidity hours
   - Reduces whipsaw during Asian session

5. Volume Confirmation:
   - Volume > 0.5x 20-bar average (very loose)
   - Ensures some participation

Key vs failed #005 (1h HMA RSI):
- LOOSER RSI thresholds (20/80 vs 30/70)
- Connors RSI instead of simple RSI (better reversal signal)
- Choppiness regime instead of rigid filters
- Volume filter at 0.5x not 1.0x

Position sizing: 0.25 (conservative for 1h TF)
Stoploss: 2.5x ATR trailing
Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 1h (target 40-80 trades/year with strict confluence)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - 3-component mean reversion indicator
    Components: RSI(3) + RSI_Streak(2) + PercentRank(100) / 3
    Proven 75% win rate for reversal entries
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of Streak Length
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on absolute streak values
    streak_abs = np.abs(streak)
    streak_gain = np.where(np.diff(streak_abs) > 0, np.diff(streak_abs), 0.0)
    streak_loss = np.where(np.diff(streak_abs) < 0, -np.diff(streak_abs), 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = avg_streak_gain[i] / (avg_streak_loss[i] + 1e-10)
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: Percent Rank of daily returns over 100 periods
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0 and np.std(returns) > 1e-10:
            current_return = close[i] - close[i-1]
            rank = np.sum(returns < current_return)
            percent_rank[i] = rank / len(returns) * 100.0
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest - lowest < 1e-10:
            chop[i] = 100.0
        else:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """Hull Moving Average - smooth and responsive"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma_full = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    wma_diff = 2.0 * wma_half - wma_full
    
    hma = pd.Series(wma_diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

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

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
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
    
    # Calculate and align 4h HMA for HTF trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative size for 1h timeframe
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        session_ok = 8 <= hour <= 20
        
        # === VOLUME FILTER (loose: > 0.5x average) ===
        volume_ok = volume[i] > 0.5 * vol_avg[i]
        
        # === 4H HMA TREND BIAS ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0  # Range market
        chop_trend = chop[i] < 45.0  # Trending market
        
        # === CONNORS RSI SIGNALS (LOOSE thresholds) ===
        crsi_oversold = crsi[i] < 25.0  # Mean reversion long
        crsi_overbought = crsi[i] > 75.0  # Mean reversion short
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: Multiple confluence (loose enough to trigger)
        # Need: Session OK + Volume OK + (HTF bullish OR range market) + CRSI oversold
        if session_ok and volume_ok:
            if hma_4h_bull and crsi_oversold:
                # Trend + pullback entry (best setup)
                desired_signal = SIZE
            elif chop_range and crsi_oversold:
                # Range market mean reversion
                desired_signal = SIZE
        
        # Short entry: Multiple confluence
        if session_ok and volume_ok:
            if hma_4h_bear and crsi_overbought:
                # Trend + pullback entry (best setup)
                desired_signal = -SIZE
            elif chop_range and crsi_overbought:
                # Range market mean reversion
                desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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
                # Flip position
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