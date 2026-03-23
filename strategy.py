#!/usr/bin/env python3
"""
Experiment #1086: 12h Primary + 1d HTF — Dual Regime (Chopiness + Connors RSI + Donchian)

Hypothesis: After 785+ failed experiments, the winning pattern for 12h timeframe is:
1. Choppiness Index (CHOP) regime detection — switches between mean-revert and trend-follow
   CHOP(14) > 61.8 = range market (use mean reversion)
   CHOP(14) < 38.2 = trending market (use breakout)
2. Connors RSI for mean reversion entries — (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 10 + price > SMA200
   Short: CRSI > 90 + price < SMA200
3. Donchian(20) breakout for trend following — break above/below 20-bar high/low
4. 1d HMA21 for macro bias — only trade in direction of higher TF trend
5. ATR(14) trailing stop 2.5x — proper risk management

Why this should beat Sharpe=0.612:
- Dual regime adapts to market conditions (major failure mode was single-regime strategies)
- Connors RSI proven 75% win rate on mean reversion (research-backed)
- Choppiness Index is best meta-filter for bear/range markets (2025 is bearish)
- 12h primary = 20-50 trades/year target (optimal for fee drag)
- 1d HTF filter prevents counter-trend trades

Timeframe: 12h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_donchian_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average — faster and smoother than EMA.
    
    Formula:
    1. WMA(period/2) * 2
    2. WMA(period) * 1
    3. Diff = (1) - (2)
    4. HMA = WMA(sqrt(period)) of Diff
    """
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_sma(series, period):
    """Simple Moving Average."""
    series = pd.Series(series)
    return series.rolling(window=period, min_periods=period).mean().values

def calculate_rsi(close, period=14):
    """
    Relative Strength Index — momentum oscillator.
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite mean reversion indicator.
    
    Formula:
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): Fast RSI on price
    RSI(streak, 2): RSI on up/down streak duration
    PercentRank(100): Percentile rank of current close vs last 100 closes
    
    CRSI < 10 = oversold (long signal)
    CRSI > 90 = overbought (short signal)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(close, 3)
    rsi_close = calculate_rsi(close, rsi_period)
    
    # RSI on streak duration
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to absolute values for RSI calculation
    streak_abs = np.abs(streak)
    streak_direction = np.sign(streak)
    
    # RSI on streak (treat streak as "price")
    rsi_streak = calculate_rsi(streak_abs + 1, streak_period)  # +1 to avoid zero
    
    # PercentRank(100) — percentile of current close vs last 100
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine into CRSI
    mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_close[mask] + rsi_streak[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures if market is trending or ranging.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/choppy market (mean reversion works)
    CHOP < 38.2 = trending market (breakout works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    range_val = highest - lowest
    mask = range_val > 1e-10
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / range_val[mask]) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel — breakout indicator.
    
    Upper = Highest High over period
    Lower = Lowest Low over period
    """
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    sma_200 = calculate_sma(close, 200)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after warmup period for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range market
        is_trending = chop[i] < 38.2  # Trending market
        
        # === MEAN REVERSION SIGNALS (Connors RSI in choppy regime) ===
        crsi_oversold = crsi[i] < 15.0  # Slightly relaxed from 10 for more trades
        crsi_overbought = crsi[i] > 85.0  # Slightly relaxed from 90
        
        # Price position filter for mean reversion
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === TREND FOLLOWING SIGNALS (Donchian breakout in trending regime) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above previous high
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # HMA trend confirmation
        hma_bull = hma_21[i] > hma_50[i]
        hma_bear = hma_21[i] < hma_50[i]
        
        # === VOLATILITY CHECK ===
        vol_spike = atr[i] > 2.0 * np.nanmedian(atr[max(0, i-100):i]) if i > 100 else False
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Mean reversion: choppy + CRSI oversold + above SMA200 + macro bull
        if is_choppy and crsi_oversold and above_sma200 and macro_bull:
            desired_signal = current_size
        
        # Trend following: trending + Donchian breakout + HMA bull + macro bull
        elif is_trending and donchian_breakout_long and hma_bull and macro_bull:
            desired_signal = current_size
        
        # === SHORT ENTRY ===
        # Mean reversion: choppy + CRSI overbought + below SMA200 + macro bear
        if is_choppy and crsi_overbought and below_sma200 and macro_bear:
            desired_signal = -current_size
        
        # Trend following: trending + Donchian breakout + HMA bear + macro bear
        elif is_trending and donchian_breakout_short and hma_bear and macro_bear:
            desired_signal = -current_size
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if regime still supports long
                if (is_choppy and crsi[i] < 50.0) or (is_trending and hma_bull):
                    if macro_bull:
                        desired_signal = current_size
            elif position_side < 0:
                # Hold short if regime still supports short
                if (is_choppy and crsi[i] > 50.0) or (is_trending and hma_bear):
                    if macro_bear:
                        desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if regime reverses or macro reverses
            if crsi[i] > 70.0:  # CRSI overbought
                desired_signal = 0.0
            if macro_bear and is_trending:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if regime reverses or macro reverses
            if crsi[i] < 30.0:  # CRSI oversold
                desired_signal = 0.0
            if macro_bull and is_trending:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals