#!/usr/bin/env python3
"""
Experiment #1465: 1h Primary + 4h/1d HTF — Relaxed Multi-Confluence with Session Filter

Hypothesis: Previous 1h strategies (#1455, #1458, #1460) failed with 0 trades due to 
OVERLY STRICT filters. This version uses PROVEN components but with RELAXED thresholds:

1. 4h HMA(21) = trend direction (proven from current best strategies)
2. 1d Choppiness(14) = regime filter but RELAXED (55/45 not 61.8/38.2)
3. Connors RSI(3,2,100) = faster than RSI(14), more entry signals
4. Session filter (8-20 UTC) = avoid low-volume Asian session
5. Volume confirmation (>0.6x avg, not 0.8x) = less restrictive
6. ATR(14) 2.5x trailing stop = risk management

Why this should work when #1455/#1458/#1460 failed:
- CRSI thresholds relaxed: <20/>80 instead of <15/>85
- Volume filter: >0.6x instead of >0.8x
- Session: 8-20 UTC (12 hours) not 10-18 UTC (8 hours)
- Multiple entry paths (4 long + 4 short) to ensure trades generate
- Position size 0.25 = conservative for 1h volatility

Target: 40-80 trades/year on 1h, Sharpe > 0.618, trades >= 30 train, >= 3 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_session_vol_4h1d_hma_atr_relaxed_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - combines RSI, streak strength, and percentile rank"""
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI component (short period for responsiveness)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:rsi_period] = np.nan
    
    # Streak RSI component
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
    
    streak_rsi = np.full(n, np.nan)
    mask = streak_loss_smooth > 1e-10
    streak_rsi[mask] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask] / streak_loss_smooth[mask]))
    streak_rsi[streak_loss_smooth <= 1e-10] = 100.0
    streak_rsi[:streak_period + 5] = np.nan
    
    # Percentile rank component
    crsi = np.full(n, np.nan)
    for i in range(rank_period, n):
        if np.isnan(rsi[i]) or np.isnan(streak_rsi[i]):
            continue
        window = close[max(0, i - rank_period + 1):i + 1]
        if len(window) >= rank_period and not np.any(np.isnan(window)):
            rank = np.sum(window < close[i]) / len(window) * 100.0
            crsi[i] = (rsi[i] + streak_rsi[i] + rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                prev_close = close[j-1] if j > 0 else close[0]
                tr_sum += max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            
            chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d Choppiness for regime
    chop_1d_raw = calculate_choppiness(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
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
        if np.isnan(crsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (>0.6x average) ===
        vol_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 1e-10 else 0.0
        vol_confirmed = vol_ratio > 0.6
        
        # === TREND FILTER (4h HMA) ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (1d Choppiness) - RELAXED ===
        is_choppy = chop_1d_aligned[i] > 55.0  # Range market
        is_trending = chop_1d_aligned[i] < 45.0  # Trend market
        
        # === CONNORS RSI EXTREMES - RELAXED ===
        crsi_oversold = crsi[i] < 20.0  # Relaxed from <15
        crsi_overbought = crsi[i] > 80.0  # Relaxed from >85
        
        # === DESIRED SIGNAL - MULTIPLE ENTRY PATHS ===
        desired_signal = 0.0
        
        # LONG ENTRIES (4 paths to ensure trades generate)
        # Path 1: Choppy regime + CRSI oversold + trend bull + session + volume
        if is_choppy and crsi_oversold and trend_bull and in_session and vol_confirmed:
            desired_signal = BASE_SIZE
        # Path 2: Trending regime + CRSI oversold + trend bull + volume
        elif is_trending and crsi_oversold and trend_bull and vol_confirmed:
            desired_signal = BASE_SIZE
        # Path 3: Very oversold CRSI + trend bull (strong mean reversion)
        elif crsi[i] < 15.0 and trend_bull:
            desired_signal = BASE_SIZE
        # Path 4: CRSI oversold + trend bull (simplified, fewer filters)
        elif crsi_oversold and trend_bull and in_session:
            desired_signal = BASE_SIZE * 0.6
        
        # SHORT ENTRIES (4 paths)
        # Path 1: Choppy regime + CRSI overbought + trend bear + session + volume
        elif is_choppy and crsi_overbought and trend_bear and in_session and vol_confirmed:
            desired_signal = -BASE_SIZE
        # Path 2: Trending regime + CRSI overbought + trend bear + volume
        elif is_trending and crsi_overbought and trend_bear and vol_confirmed:
            desired_signal = -BASE_SIZE
        # Path 3: Very overbought CRSI + trend bear (strong mean reversion)
        elif crsi[i] > 85.0 and trend_bear:
            desired_signal = -BASE_SIZE
        # Path 4: CRSI overbought + trend bear (simplified, fewer filters)
        elif crsi_overbought and trend_bear and in_session:
            desired_signal = -BASE_SIZE * 0.6
        
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
        if desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
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