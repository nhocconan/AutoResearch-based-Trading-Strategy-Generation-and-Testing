#!/usr/bin/env python3
"""
Experiment #1280: 1h Primary + 4h/12h HTF — Regime-Adaptive CRSI with Session Filter

Hypothesis: Recent failures (#1268-1279) mostly have Sharpe=0.000 = ZERO TRADES.
Entry conditions TOO STRICT. This strategy uses:
1. 1h primary timeframe with 4h/12h HTF for direction (proven to reduce whipsaw)
2. CHOPPINESS INDEX regime (CHOP>50=range, CHOP<50=trend) — NO GAP between thresholds
3. CONNORS RSI with LOOSE bands (CRSI<30 long, >70 short) — ensures trades
4. 4h HMA for intermediate trend, 12h HMA for macro filter
5. Session filter (8-20 UTC) — only trade during liquid hours
6. Volume filter (>0.7x 20-bar avg) — achievable threshold
7. BASE_SIZE=0.25 (smaller for 1h to reduce fee drag)

Key changes from failed #1275:
- Remove ADX requirement (was blocking signals)
- Widen CRSI bands (30/70 vs 20/80)
- CHOP threshold at 50 (no gap between range/trend)
- Lower volume threshold (0.7x vs 1.0x)
- Ensure minimum 30 trades/year by loosening confluence

Target: Sharpe > 0.612, trades >= 40 train, >= 5 test, DD < -30%
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_4h12h_hma_session_vol_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection
    CHOP > 50 = ranging (mean revert)
    CHOP < 50 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh > ll and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - mean reversion signal
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 30 (loose threshold for more trades)
    Short: CRSI > 70
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    mask = loss_smooth > 1e-10
    rsi_short[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi_short[:rsi_period] = np.nan
    
    # RSI Streak (2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    mask2 = streak_loss_smooth > 1e-10
    rsi_streak[mask2] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask2] / streak_loss_smooth[mask2]))
    rsi_streak[:streak_period] = np.nan
    
    # Percent Rank (100)
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        pct_rank[i] = 100.0 * rank / (rank_period - 1)
    pct_rank[:rank_period] = np.nan
    
    # Combine
    valid = (~np.isnan(rsi_short)) & (~np.isnan(rsi_streak)) & (~np.isnan(pct_rank))
    crsi[valid] = (rsi_short[valid] + rsi_streak[valid] + pct_rank[valid]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return mid, upper, lower
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        mid[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = mid[i] + std_mult * std
        lower[i] = mid[i] - std_mult * std
    
    return mid, upper, lower

def calculate_volume_avg(volume, period=20):
    """Rolling average volume"""
    n = len(volume)
    vol_avg = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        vol_avg[i] = np.mean(volume[i-period+1:i+1])
    
    return vol_avg

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds)"""
    hours = np.zeros(len(open_time_array), dtype=int)
    for i in range(len(open_time_array)):
        # open_time is in milliseconds since epoch
        hours[i] = (open_time_array[i] // (1000 * 3600)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA for trend filters
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    vol_avg = calculate_volume_avg(volume, period=20)
    utc_hour = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller for 1h to reduce fee drag
    
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
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(bb_mid[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        # === VOLUME FILTER (>0.7x average) ===
        volume_ok = volume[i] >= 0.7 * vol_avg[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        in_range = chop[i] >= 50.0  # Ranging market
        in_trend = chop[i] < 50.0   # Trending market
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        hma_bull = close[i] > hma_4h_aligned[i]
        hma_bear = close[i] < hma_4h_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Follow trend on pullback
        if in_trend:
            # Long: Macro bull + 4h bull + session + volume
            if macro_bull and hma_bull and in_session and volume_ok:
                # Enter on CRSI pullback (not too deep, just < 50)
                if crsi[i] < 50.0:
                    desired_signal = BASE_SIZE
            # Short: Macro bear + 4h bear + session + volume
            elif macro_bear and hma_bear and in_session and volume_ok:
                if crsi[i] > 50.0:
                    desired_signal = -BASE_SIZE
        
        # RANGING REGIME: Mean revert with CRSI extremes
        elif in_range:
            # Long: CRSI oversold + price near BB lower
            if crsi[i] < 30.0 and close[i] <= bb_lower[i] * 1.005:
                if in_session and volume_ok:
                    desired_signal = BASE_SIZE
            # Short: CRSI overbought + price near BB upper
            elif crsi[i] > 70.0 and close[i] >= bb_upper[i] * 0.995:
                if in_session and volume_ok:
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
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