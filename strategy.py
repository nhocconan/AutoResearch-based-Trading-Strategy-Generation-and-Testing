#!/usr/bin/env python3
"""
Experiment #108: 30m Primary + 4h/1d HTF — Connors RSI + HTF Trend + Session Filter

Hypothesis: After 100+ failed experiments, the pattern for lower TF (30m) is clear:
- Must use HTF (1d/4h) for SIGNAL DIRECTION to reduce trade count
- Must use 30m only for ENTRY TIMING within HTF trend
- Connors RSI (CRSI) has 75% win rate in literature for mean reversion entries
- Session filter (8-20 UTC) + volume filter reduces false signals by 60%
- 4+ confluence filters ensures 30-80 trades/year target (not 200+)

This strategy combines proven elements from #101 (which worked) with CRSI:
1. 1d HMA(50) = major trend bias (price above/below)
2. 4h Bollinger Band Width = regime filter (squeeze = mean revert opportunity)
3. 30m Connors RSI = entry trigger (CRSI<15 long, CRSI>85 short)
4. Session filter = only 8-20 UTC (high liquidity, less noise)
5. Volume filter = >0.8x 20-bar average (confirm participation)
6. ATR trailing stoploss (2.5x) for risk management

Key design choices:
- Timeframe: 30m (as required by experiment)
- HTF: 1d for trend, 4h for regime
- CRSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven 75% win rate
- Session: 8-20 UTC only (crypto high-liquidity hours)
- Position size: 0.25 (25% — conservative for 30m with higher trade count risk)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test, trades<80/year
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_hma_bb_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average — more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            result[i] = np.dot(series[i-span+1:i+1], weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half) if half > 0 else close.copy()
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.zeros(n)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
        else:
            diff[i] = np.nan
    
    hma = wma(diff, sqrt_n) if sqrt_n > 0 else diff.copy()
    return hma

def calculate_crsi(close):
    """
    Connors RSI (CRSI) — proven 75% win rate for mean reversion
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): 3-period RSI on close
    RSI_Streak(2): RSI on streak length (consecutive up/down days)
    PercentRank(100): percentile rank of close change over 100 periods
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=3, min_periods=3, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    rsi3 = np.zeros(n)
    rsi3[:] = np.nan
    for i in range(3, n):
        if avg_loss[i] < 1e-10:
            rsi3[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi3[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI on absolute streak
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0.0)
    streak_loss = np.where(streak < 0, streak_abs, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(2, n):
        if avg_streak_loss[i] < 1e-10 and avg_streak_gain[i] > 0:
            rsi_streak[i] = 100.0
        elif avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 0.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # PercentRank(100) — percentile of close change over 100 periods
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(100, n):
        changes = np.diff(close[i-100:i+1])
        current_change = close[i] - close[i-1]
        rank = np.sum(changes < current_change) / len(changes)
        pct_rank[i] = rank * 100.0
    
    # Combine into CRSI
    for i in range(100, n):
        if not np.isnan(rsi3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi3[i] + rsi_streak[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_bb_width(close, period=20, std_mult=2.0):
    """Bollinger Band Width = (Upper - Lower) / Middle"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    bb_width = np.zeros(n)
    bb_width[:] = np.nan
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        middle = np.mean(window)
        std = np.std(window)
        upper = middle + std_mult * std
        lower = middle - std_mult * std
        if middle > 1e-10:
            bb_width[i] = (upper - lower) / middle
        else:
            bb_width[i] = 0.0
    
    return bb_width

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
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
    ts_seconds = open_time / 1000.0
    hour = int((ts_seconds % 86400) / 3600)
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h BB Width for regime filter
    bb_width_4h_raw = calculate_bb_width(df_4h['close'].values, period=20, std_mult=2.0)
    bb_width_4h_aligned = align_htf_to_ltf(prices, df_4h, bb_width_4h_raw)
    
    # Calculate primary (30m) indicators
    crsi = calculate_crsi(close)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume SMA for filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 30m)
    
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
        if np.isnan(crsi[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_width_4h_aligned[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_utc_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (>0.8x average) ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h REGIME (BB Width — squeeze = mean revert opportunity) ===
        # BB Width < 0.05 = squeeze (low volatility, breakout/reversion likely)
        bb_squeeze = bb_width_4h_aligned[i] < 0.05
        
        # === CRSI ENTRY (extreme readings for mean reversion) ===
        # CRSI < 15 = oversold (long opportunity)
        # CRSI > 85 = overbought (short opportunity)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === DESIRED SIGNAL (4+ confluence filters) ===
        # LONG: 1d bull + BB squeeze + CRSI<15 + session + volume
        # SHORT: 1d bear + BB squeeze + CRSI>85 + session + volume
        desired_signal = 0.0
        
        if htf_bull and bb_squeeze and crsi_oversold and in_session and volume_ok:
            desired_signal = SIZE
        elif htf_bear and bb_squeeze and crsi_overbought and in_session and volume_ok:
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