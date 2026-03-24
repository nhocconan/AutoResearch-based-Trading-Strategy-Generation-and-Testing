#!/usr/bin/env python3
"""
Experiment #1623: 1d Primary + 1w HTF — Dual Regime (Chop/Trend) + CRSI + Donchian

Hypothesis: After analyzing 1200+ failed strategies, the key insight is REGIME ADAPTATION.
BTC/ETH fail with pure trend following (2022 crash whipsaw destroys gains). Pure mean
reversion fails in strong trends. The solution: detect regime with Choppiness Index,
then apply the RIGHT strategy for that regime.

Key components:
1. CHOP(14) regime filter: >61.8 = chop (mean revert), <38.2 = trend (breakout)
2. CHOPPY regime: Connors RSI for mean reversion (CRSI<15 long, >85 short)
3. TREND regime: Donchian(20) breakout with 1w HMA bias filter
4. 1w HMA for ultra-long-term bias (prevents counter-trend in major moves)
5. ATR(14) trailing stop at 2.5x for risk management
6. Position size: 0.28 (conservative, allows survival through 2022 crash)

Why this should beat Sharpe 0.618:
- Regime adaptation works in both bull (2021) and bear (2022, 2025) markets
- CRSI mean reversion has 75% win rate in range markets (research-backed)
- Donchian breakouts capture major trends without whipsaw in chop
- 1w HMA bias prevents fatal counter-trend positions
- 1d timeframe targets 20-50 trades/year (optimal fee/trade balance)
- Looser CRSI thresholds (15/85 vs 10/90) ensures sufficient trades

Timeframe: 1d (required for this experiment)
HTF: 1w HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_crsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    CHOP > 61.8 = range/chop, CHOP < 38.2 = trend
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    CRSI < 15 = oversold (long), CRSI > 85 = overbought (short)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(close, 3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi_close[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi_close[loss_smooth <= 1e-10] = 100.0
    rsi_close[:rsi_period] = np.nan
    
    # RSI(streak, 2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask2 = streak_loss_smooth > 1e-10
    rsi_streak[mask2] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask2] / streak_loss_smooth[mask2]))
    rsi_streak[streak_loss_smooth <= 1e-10] = 100.0
    rsi_streak[:streak_period] = np.nan
    
    # PercentRank(close, 100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1)
        percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_close[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - breakout indicator
    Returns: upper (highest high), lower (lowest low), middle
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for ultra-long-term bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Donchian channels for breakout detection
    donch_upper, donch_lower, donch_middle = calculate_donchian(high, low, period=20)
    
    # SMA for additional trend filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
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
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range market (slightly lower threshold for more trades)
        is_trending = chop[i] < 45.0  # Trend market
        
        # === TREND BIAS (1w HMA + SMA) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # Additional trend confirmation
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 20.0  # LOOSE for more trades (was 15)
        crsi_overbought = crsi[i] > 80.0  # LOOSE for more trades (was 85)
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donch_upper[i-1] if not np.isnan(donch_upper[i-1]) else False
        breakout_short = close[i] < donch_lower[i-1] if not np.isnan(donch_lower[i-1]) else False
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY MARKET - Mean Reversion with CRSI
        if is_choppy:
            # Mean reversion: buy oversold, sell overbought
            # Only enter with weekly bias support (safer)
            if crsi_oversold and weekly_bull:
                desired_signal = BASE_SIZE
            elif crsi_overbought and weekly_bear:
                desired_signal = -BASE_SIZE
            # Secondary: CRSI extreme even without weekly confirmation (more trades)
            elif crsi[i] < 10.0:
                desired_signal = BASE_SIZE
            elif crsi[i] > 90.0:
                desired_signal = -BASE_SIZE
        
        # REGIME 2: TRENDING MARKET - Breakout with Donchian + Weekly Bias
        elif is_trending:
            # Long breakout only in bull market
            if breakout_long and weekly_bull:
                desired_signal = BASE_SIZE
            # Short breakout only in bear market
            elif breakout_short and weekly_bear:
                desired_signal = -BASE_SIZE
            # Secondary: breakout with SMA confirmation (more trades)
            elif breakout_long and above_sma50:
                desired_signal = BASE_SIZE
            elif breakout_short and not above_sma50:
                desired_signal = -BASE_SIZE
        
        # REGIME 3: NEUTRAL/TRANSITION - Hold existing or wait
        else:
            # In neutral regime, only hold existing positions
            # But allow entries on extreme CRSI for more trades
            if crsi[i] < 8.0:
                desired_signal = BASE_SIZE
            elif crsi[i] > 92.0:
                desired_signal = -BASE_SIZE
            elif in_position:
                desired_signal = BASE_SIZE if position_side > 0 else -BASE_SIZE
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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