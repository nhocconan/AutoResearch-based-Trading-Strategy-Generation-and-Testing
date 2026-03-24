#!/usr/bin/env python3
"""
Experiment #619: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + CRSI Mean Reversion

Hypothesis: 1h timeframe with 4h/12h HTF trend filter provides optimal balance between
trade frequency (40-80/year) and signal quality. Key improvements over failed #610/618:

1. SIMPLER entry logic - fewer confluence requirements = MORE trades (avoid 0-trade failure)
2. OR logic for entries - trend pullback OR mean reversion (not both required)
3. Session filter 08-20 UTC for NEW entries only (don't force exit existing positions)
4. 4h HMA(21) + 12h HMA(21) for trend direction (proven reliable, simpler than KAMA)
5. 1h RSI(14) pullback entries in trend direction (RSI 35-55 for long, 45-65 for short)
6. CRSI extremes for additional mean reversion signals (CRSI<20 long, >80 short)
7. ATR(14)*2.5 stoploss with trailing

Strategy logic:
1. 12h HMA(21) = macro trend bias
2. 4h HMA(21) = medium trend bias  
3. 1h RSI(14) = entry timing (pullback to 35-55 in uptrend, 45-65 in downtrend)
4. 1h CRSI(3,2,100) = extreme mean reversion (<20 long, >80 short)
5. Session: 08-20 UTC for NEW entries only (avoid Asia low liquidity)
6. ATR(14)*2.5 stoploss with trailing

Entry conditions (ANY triggers - OR logic for more trades):
- TREND LONG: 4h HMA bull + 12h HMA bull + RSI 35-55 + session
- TREND SHORT: 4h HMA bear + 12h HMA bear + RSI 45-65 + session
- MEAN REV LONG: CRSI<20 + RSI<40
- MEAN REV SHORT: CRSI>80 + RSI>60

Target: Sharpe>0.40, trades>=40 train, trades>=5 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_crsi_4h12h_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(close,3) + RSI(Streak,2) + PercentRank(100)) / 3
    
    CRSI < 20 = extreme oversold (long signal)
    CRSI > 80 = extreme overbought (short signal)
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 1 and close[i-1] >= close[i-2] else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 1 and close[i-1] <= close[i-2] else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

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

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for medium trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC for NEW entries only) ===
        open_time = prices["open_time"].iloc[i]
        hour = (open_time // 3600000) % 24
        in_session = 8 <= hour <= 20
        
        # === HTF BIAS (12h macro + 4h medium) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_12h_aligned[i]
        
        # === RSI CONDITIONS ===
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi[i] <= 65.0
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === ENTRY LOGIC (OR logic - any condition triggers) ===
        desired_signal = 0.0
        
        # TREND PULLBACK LONG: HTF bull + RSI pullback + session
        if htf_bull and rsi_pullback_long and in_session:
            desired_signal = SIZE_BASE
        
        # TREND PULLBACK SHORT: HTF bear + RSI pullback + session
        elif htf_bear and rsi_pullback_short and in_session:
            desired_signal = -SIZE_BASE
        
        # MEAN REVERSION LONG: CRSI extreme + RSI oversold
        elif crsi_oversold and rsi_oversold:
            desired_signal = SIZE_BASE
        
        # MEAN REVERSION SHORT: CRSI extreme + RSI overbought
        elif crsi_overbought and rsi_overbought:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals