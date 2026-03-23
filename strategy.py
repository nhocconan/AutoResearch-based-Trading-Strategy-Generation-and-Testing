#!/usr/bin/env python3
"""
Experiment #1088: 30m Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + HTF HMA Trend

Hypothesis: 30m timeframe can work IF we use extreme selectivity:
1. 1d HMA21 for macro bias (only trade with daily trend)
2. 4h HMA21 for intermediate trend confirmation
3. Choppiness Index(14) regime filter: CHOP>55 = range (mean revert), CHOP<45 = trend
4. Connors RSI for entries: CRSI<15 long, CRSI>85 short (extreme only)
5. Session filter: only 8-20 UTC (high liquidity)
6. Volume filter: volume > 0.8x 20-bar average
7. ATR trailing stop 2.5x

This should generate 30-60 trades/year (not 200+) with higher win rate.
Position Size: 0.20-0.25 (smaller for lower TF to reduce fee impact)

Why this differs from failed #1078:
- Stricter CRSI thresholds (15/85 vs 40/60) = fewer but higher quality trades
- Added Choppiness regime filter = avoids trend strategies in chop
- Dual HTF (4h + 1d) = stronger trend confirmation
- Session filter = avoids low liquidity whipsaws
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_dual_htf_hma_session_vol_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average — faster and smoother than EMA.
    Formula: HMA = WMA(sqrt(period)) of (2*WMA(period/2) - WMA(period))
    """
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    Formula: CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range
    CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    denom = hh - ll
    mask = denom > 1e-10
    chop[mask] = 100.0 * np.log10(tr_sum[mask] / denom[mask]) / np.log10(period)
    chop[~mask] = 50.0
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — combines 3 components for mean reversion signals.
    Formula: CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    CRSI < 10 = extreme oversold (long)
    CRSI > 90 = extreme overbought (short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # RSI on streak (consecutive up/down days)
    diff = np.diff(close)
    streak = np.zeros(n)
    for i in range(1, n):
        if diff[i-1] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif diff[i-1] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, 50.0)
    mask = avg_streak_loss > 1e-10
    rs_streak = np.zeros(n)
    rs_streak[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + rs_streak[mask]))
    
    # Percent Rank of close over lookback
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        lookback = close[i-rank_period:i]
        current = close[i]
        count_below = np.sum(lookback < current)
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    mask_valid = (~np.isnan(rsi_close)) & (~np.isnan(rsi_streak)) & (~np.isnan(percent_rank))
    crsi[mask_valid] = (rsi_close[mask_valid] + rsi_streak[mask_valid] + percent_rank[mask_valid]) / 3.0
    
    return crsi

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

def get_hour_from_open_time(open_time):
    """Extract hour from Binance open_time (milliseconds timestamp)."""
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA21 for trend filters
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume average for filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(rsi[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or atr[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === MACRO TREND (1d HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA21) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY TREND (30m HMA crossover) ===
        hma_bull = hma_21[i] > hma_50[i]
        hma_bear = hma_21[i] < hma_50[i]
        
        # === CHOPPINESS REGIME ===
        choppy_regime = chop[i] > 55.0  # Range market
        trending_regime = chop[i] < 45.0  # Trending market
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY (3+ confluence required) ===
        # Must have: session + volume + (macro OR 4h bull) + (crsi extreme OR hma bull in trend regime)
        long_conditions = [
            in_session,
            volume_ok,
            (macro_bull or hma_4h_bull),  # At least one HTF bullish
        ]
        
        # Add trend/momentum confirmation
        if trending_regime:
            # In trending regime: need HMA bullish + CRSI not overbought
            if hma_bull and crsi[i] < 70.0:
                long_conditions.append(True)
            else:
                long_conditions.append(False)
        else:
            # In choppy regime: need CRSI extreme oversold for mean reversion
            if crsi_oversold:
                long_conditions.append(True)
            else:
                long_conditions.append(False)
        
        if sum(long_conditions) >= 4:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY (3+ confluence required) ===
        short_conditions = [
            in_session,
            volume_ok,
            (macro_bear or hma_4h_bear),  # At least one HTF bearish
        ]
        
        if trending_regime:
            # In trending regime: need HMA bearish + CRSI not oversold
            if hma_bear and crsi[i] > 30.0:
                short_conditions.append(True)
            else:
                short_conditions.append(False)
        else:
            # In choppy regime: need CRSI extreme overbought for mean reversion
            if crsi_overbought:
                short_conditions.append(True)
            else:
                short_conditions.append(False)
        
        if sum(short_conditions) >= 4:
            desired_signal = -BASE_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish and CRSI not extreme
                if (macro_bull or hma_4h_bull) and crsi[i] < 80.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HTF still bearish and CRSI not extreme
                if (macro_bear or hma_4h_bear) and crsi[i] > 20.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both HTF reverse or CRSI extreme overbought
            if macro_bear and hma_4h_bear:
                desired_signal = 0.0
            if crsi_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both HTF reverse or CRSI extreme oversold
            if macro_bull and hma_4h_bull:
                desired_signal = 0.0
            if crsi_oversold:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
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