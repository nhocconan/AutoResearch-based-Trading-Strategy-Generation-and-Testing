#!/usr/bin/env python3
"""
Experiment #504: 12h Primary + 1d/1w HTF — Dual Regime Trend/Mean Reversion

Hypothesis: 12h timeframe with dual HTF (1d + 1w) provides optimal signal quality.
Use 1w HMA for long-term bias, 1d HMA for medium-term trend, 12h indicators for entry.
Dual regime: trend-follow when HTF aligned, mean-revert on RSI extremes otherwise.
Connors RSI (CRSI) for mean reversion entries (proven on ETH with Sharpe +0.923).

Strategy logic:
1. 1w HMA(21) = long-term bias (slowest HTF)
2. 1d HMA(21) = medium-term trend (faster HTF)
3. 12h RSI(14) + CRSI(3,2,100) = entry timing
4. 12h Donchian(20) = breakout confirmation
5. 12h ATR(14)*2.5 stoploss on all positions
6. OR logic for entries to ensure trade generation

Key improvements from failed experiments:
- Dual HTF (1d + 1w) for better regime detection
- Connors RSI for mean reversion (proven edge)
- Loose entry thresholds to ensure trades generated
- Simple logic, no complex regime switching

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=15 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_hma_crsi_donchian_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_abs = abs(streak[i])
        if streak[i] > 0:
            streak_rsi[i] = 50.0 + min(streak_abs * 10.0, 50.0)
        elif streak[i] < 0:
            streak_rsi[i] = 50.0 - min(streak_abs * 10.0, 50.0)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        if len(window) >= rank_period:
            count_below = np.sum(window[:-1] < close[i])
            percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    plus_di[:] = np.nan
    minus_di[:] = np.nan
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for long-term bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for medium-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    hma_12h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w HTF LONG-TERM BIAS ===
        htf_weekly_bull = close[i] > hma_1w_aligned[i]
        htf_weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d HTF MEDIUM-TERM TREND ===
        htf_daily_bull = close[i] > hma_1d_aligned[i]
        htf_daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i] if not np.isnan(sma_50[i]) else False
        below_sma50 = close[i] < sma_50[i] if not np.isnan(sma_50[i]) else False
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === RSI/CRSI EXTREMES (LOOSE: 35/65 for entries) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_extreme_oversold = rsi[i] < 35.0
        rsi_extreme_overbought = rsi[i] > 65.0
        
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 25.0
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 75.0
        crsi_extreme_oversold = not np.isnan(crsi[i]) and crsi[i] < 15.0
        crsi_extreme_overbought = not np.isnan(crsi[i]) and crsi[i] > 85.0
        
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = not np.isnan(adx[i]) and adx[i] > 25.0
        adx_weak = not np.isnan(adx[i]) and adx[i] < 20.0
        
        # === VOLATILITY FILTER ===
        atr_ratio = atr[i] / np.nanmean(atr[max(0,i-100):i]) if i >= 100 else 1.0
        vol_normal = atr_ratio < 2.5
        
        # === ENTRY LOGIC (LOOSE - OR logic, not AND) ===
        desired_signal = 0.0
        
        # TREND LONG: HTF aligned bull + breakout/cross
        if htf_weekly_bull and htf_daily_bull and vol_normal:
            if donchian_breakout_long and above_sma50:
                desired_signal = SIZE_STRONG
            elif hma_bull and rsi[i] > 50.0 and rsi_rising and above_sma50:
                desired_signal = SIZE_BASE
            elif rsi_extreme_oversold and rsi_rising and above_sma50:
                desired_signal = SIZE_BASE
            elif crsi_extreme_oversold and above_sma50:
                desired_signal = SIZE_BASE
        
        # TREND SHORT: HTF aligned bear + breakdown/cross
        elif htf_weekly_bear and htf_daily_bear and vol_normal:
            if donchian_breakdown_short and below_sma50:
                desired_signal = -SIZE_STRONG
            elif hma_bear and rsi[i] < 50.0 and rsi_falling and below_sma50:
                desired_signal = -SIZE_BASE
            elif rsi_extreme_overbought and rsi_falling and below_sma50:
                desired_signal = -SIZE_BASE
            elif crsi_extreme_overbought and below_sma50:
                desired_signal = -SIZE_BASE
        
        # MEAN REVERSION LONG: CRSI/RSI extreme (works in any HTF regime)
        if desired_signal == 0.0 and vol_normal:
            if crsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE
            elif crsi_oversold and above_sma50:
                desired_signal = SIZE_BASE * 0.8
            elif rsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE
            elif rsi_oversold and above_sma50 and rsi_rising:
                desired_signal = SIZE_BASE * 0.8
        
        # MEAN REVERSION SHORT: CRSI/RSI extreme (works in any HTF regime)
        if desired_signal == 0.0 and vol_normal:
            if crsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            elif crsi_overbought and below_sma50:
                desired_signal = -SIZE_BASE * 0.8
            elif rsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            elif rsi_overbought and below_sma50 and rsi_falling:
                desired_signal = -SIZE_BASE * 0.8
        
        # DONCHIAN BREAKOUT: Strong momentum signal (any regime)
        if desired_signal == 0.0 and vol_normal:
            if donchian_breakout_long and adx_strong and rsi[i] > 45.0:
                desired_signal = SIZE_BASE * 0.8
            elif donchian_breakdown_short and adx_strong and rsi[i] < 55.0:
                desired_signal = -SIZE_BASE * 0.8
        
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
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