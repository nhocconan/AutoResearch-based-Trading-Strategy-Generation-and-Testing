#!/usr/bin/env python3
"""
Experiment #692: 12h Primary + 1d/1w HTF — Donchian Breakout + HMA Trend + CRSI Filter

Hypothesis: After analyzing 691 failed strategies, the pattern for 12h success is:
1. #682 (12h chop crsi hma 1d) failed with Sharpe=-0.206 — too many regime filters
2. Donchian breakout worked on SOL (Sharpe +0.782 in research notes)
3. 12h needs SIMPLER logic than 4h — fewer filters, more trades
4. Current best (1d Chop+CRSI+1w) has Sharpe=0.520 — we need different edge

This strategy uses:
- Donchian(20) breakout for trend entry (proven on crypto)
- 1d HMA for major trend bias (simpler, faster than 1w)
- 1w HMA for secular trend filter (only trade with weekly trend)
- Connors RSI(3,2,100) for pullback entries within trend
- NO Choppiness filter (caused 0 trades in #682, #685)

Why this might beat Sharpe=0.520:
- 12h timeframe = 15-30 trades/year (optimal per Rule 10)
- Donchian breakout catches major moves without whipsaw
- Dual HTF (1d + 1w) ensures we trade with major trend
- CRSI filter prevents chasing breakouts at extremes
- Conservative sizing (0.25) + ATR stop controls drawdown

Position sizing: 0.25 discrete (per Rule 4, max 0.40)
Target: 20-40 trades/year on 12h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_crsi_1d1w_v1"
timeframe = "12h"
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
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel.
    Upper = Highest High over period
    Lower = Lowest Low over period
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
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
    
    # PercentRank(100)
    percent_rank = pd.Series(returns).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=False
    ).values * 100.0
    
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF HMAs
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_12h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS (secular trend) ===
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-10] if i >= 10 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-10] if i >= 10 else False
        
        # === 1D TREND BIAS (intermediate trend) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 12H HMA SLOPE ===
        hma_12h_slope_bull = hma_12h[i] > hma_12h[i-3] if i >= 3 else False
        hma_12h_slope_bear = hma_12h[i] < hma_12h[i-3] if i >= 3 else False
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous high
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === CONNORS RSI FILTER (prevent chasing) ===
        crsi_neutral = 30.0 < crsi[i] < 70.0  # Not at extreme
        crsi_oversold = crsi[i] < 40.0  # Pullback opportunity
        crsi_overbought = crsi[i] > 60.0  # Rally opportunity
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Condition 1: Weekly bull + Daily bull + Donchian breakout + CRSI not overbought
        if hma_1w_slope_bull and hma_1d_slope_bull:
            if breakout_long and crsi_oversold:
                new_signal = POSITION_SIZE
            # Condition 2: Pullback entry in uptrend
            elif price_above_hma_1d and hma_12h_slope_bull and crsi[i] < 35.0:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Condition 1: Weekly bear + Daily bear + Donchian breakout + CRSI not oversold
        elif hma_1w_slope_bear and hma_1d_slope_bear:
            if breakout_short and crsi_overbought:
                new_signal = -POSITION_SIZE
            # Condition 2: Pullback entry in downtrend
            elif price_below_hma_1d and hma_12h_slope_bear and crsi[i] > 65.0:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
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