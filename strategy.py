#!/usr/bin/env python3
"""
Experiment #137: 1d Primary + 1w HTF — Connors RSI Mean Reversion + HMA Trend Filter

Hypothesis: After 136 failed experiments, the pattern is clear:
- Complex multi-regime filters cause 0 trades (experiments #125, #127, #135)
- Connors RSI on 1d showed ETH Sharpe +0.923 in research literature
- 1d timeframe with 1w HTF bias reduces noise while maintaining trade frequency
- Mean reversion works better than trend-following in 2022-2024 mixed market

This strategy combines:
1. Connors RSI (CRSI) for mean reversion entries — proven 75% win rate
2. 1w HMA(21) for major trend bias — only trade with weekly trend
3. ADX(14) > 12 filter — avoid completely dead markets
4. ATR(14) 2.5x trailing stoploss for risk management
5. NO complex regime detection, NO Choppiness Index

Key design choices:
- Timeframe: 1d (as required by experiment, 20-50 trades/year target)
- HTF: 1w for trend bias (stable, less whipsaw than 1d)
- CRSI thresholds: 20/80 (looser than 15/85 to ensure trade generation)
- ADX threshold: 12 (minimum momentum, not too restrictive)
- Position size: 0.30 (30% of capital, conservative for 1d)
- Stoploss: 2.5x ATR trailing (proven in baseline strategy)

Target: Sharpe>0.351 (beat current best), DD>-40%, trades>=80 on train, trades>=15 on test
All symbols (BTC, ETH, SOL) must have Sharpe>0 individually.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_hma_meanrev_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_lookback=100):
    """
    Connors RSI (CRSI) — Larry Connors' mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    - RSI(3): Short-term momentum
    - RSI_Streak(2): RSI of consecutive up/down days
    - PercentRank(100): Where current price ranks vs last 100 days
    
    Entry signals: CRSI < 20 (oversold long), CRSI > 80 (overbought short)
    Research shows 75% win rate on daily timeframe.
    """
    n = len(close)
    if n < rank_lookback + 5:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_3 = np.zeros(n)
    rsi_3[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_3[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_3[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: Streak RSI(2)
    # Streak = consecutive up days (positive) or down days (negative)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if streak_avg_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = streak_avg_gain[i] / streak_avg_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_lookback, n):
        window = close[i-rank_lookback:i+1]
        current = close[i]
        count_below = np.sum(window < current)
        percent_rank[i] = 100.0 * count_below / rank_lookback
    
    # Combine components
    for i in range(rank_lookback, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    
    Advantages over EMA/SMA:
    - Reduces lag significantly
    - Smoother than DEMA
    - Better trend identification for HTF bias
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA using EMA approximation (close enough for HMA purposes)
    wma_half = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma_full = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Hull input
    hull_input = 2.0 * wma_half - wma_full
    
    # Final smoothing
    hma = pd.Series(hull_input).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength, not direction.
    ADX > 25 = trending, ADX < 20 = ranging
    We use ADX > 12 as minimum filter to avoid dead markets.
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth TR, +DM, -DM
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / atr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_lookback=100)
    hma_1d = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 1d)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after CRSI warmup (100) + ADX warmup (28)
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_1d[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        # Only trade in direction of weekly trend
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === CRSI EXTREMES (Mean Reversion) ===
        # LOOSE thresholds to ensure trade generation on all symbols
        crsi_oversold = crsi[i] < 20.0  # Long entry
        crsi_overbought = crsi[i] > 80.0  # Short entry
        
        # === ADX FILTER (ensure some momentum) ===
        # Minimum threshold to avoid completely dead/choppy markets
        adx_ok = adx[i] > 12.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1w bull + CRSI oversold + ADX > 12
        # SHORT: 1w bear + CRSI overbought + ADX > 12
        desired_signal = 0.0
        
        if htf_bull and crsi_oversold and adx_ok:
            desired_signal = SIZE
        elif htf_bear and crsi_overbought and adx_ok:
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