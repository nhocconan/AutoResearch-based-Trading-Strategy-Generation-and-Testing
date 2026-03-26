#!/usr/bin/env python3
"""
Experiment #021: 12h CRSI(9) + Donchian Channel + Volume Spike

HYPOTHESIS: 12h timeframe reduces noise vs 4h while remaining actionable.
CRSI(9) is a proven edge from the DB (test Sharpe 1.46 on SOL). It combines
RSI(3) + RSI streak + percent rank for more responsive overbought/oversold.
Donchian(20) breakout confirms the move has momentum. Volume spike (>1.2x MA20)
filters false breakouts. Works in both bull (long CRSI<20 + breakout + vol) 
and bear (short CRSI>80 + breakdown + vol).

TIMEFRAME: 12h primary
HTF: 1d for trend bias (filter counter-trend trades)
TARGET: 75-200 total trades over 4 years (19-50/year)
SIZE: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_donchian_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, period=9, roc_period=2, rank_period=100):
    """
    Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    More responsive than standard RSI, better for shorter lookbacks.
    """
    n = len(close)
    if n < max(period, rank_period):
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain_3 = gain.ewm(span=3, min_periods=3, adjust=False).mean()
    avg_loss_3 = loss.ewm(span=3, min_periods=3, adjust=False).mean()
    rs_3 = avg_gain_3 / (avg_loss_3 + 1e-10)
    rsi_3 = (100 - (100 / (1 + rs_3))).values
    
    # Component 2: RSI Streak (2-period)
    streak = np.zeros(n)
    streak[0] = 0
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + delta.iloc[i]
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] + delta.iloc[i]
        else:
            streak[i] = 0
    
    streak_series = pd.Series(streak)
    gain_s = streak_series.where(streak_series > 0, 0.0)
    loss_s = (-streak_series).where(streak_series < 0, 0.0)
    avg_gain_s = gain_s.ewm(span=2, min_periods=2, adjust=False).mean()
    avg_loss_s = loss_s.ewm(span=2, min_periods=2, adjust=False).mean()
    rs_s = avg_gain_s / (avg_loss_s + 1e-10)
    rsi_streak = (100 - (100 / (1 + rs_s))).values
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, 50.0)
    for i in range(rank_period - 1, n):
        window = close[max(0, i - rank_period + 1):i + 1]
        rank = np.sum(window < close[i])
        percent_rank[i] = 100.0 * rank / len(window)
    
    # Combined CRSI
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, middle, lower

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === Pre-compute all indicators (vectorized) ===
    # CRSI(9) - main signal
    crsi = calculate_crsi(close, period=9)
    
    # ATR(14) for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian(20)
    donch_upper, donch_mid, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA20 and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Simple RSI(14) for additional confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(100, rank_period for _ in [1])  # ensure CRSI is warm
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 0:
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
        
        # Get 1d trend (use last known value)
        trend_bullish = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        
        # === CONDITIONS ===
        crsi_val = crsi[i]
        vol_spike = vol_ratio[i] > 1.2  # loose enough for 12h
        rsi_val = rsi_14[i]
        
        # Donchian breakout detection (cross above/below)
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # Previous bar prices for crossover detection
        prev_above_upper = close[i-1] > donch_upper[i-1] if i > 1 else False
        prev_below_lower = close[i-1] < donch_lower[i-1] if i > 1 else False
        
        # Breakout: price crosses outside channel
        breakout_up = price_above_upper and not prev_above_upper
        breakout_down = price_below_lower and not prev_below_lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: CRSI < 20 (oversold) + volume spike + bullish 1d trend
            # OR breakout up with volume
            if crsi_val < 20 and vol_spike and trend_bullish:
                desired_signal = SIZE
            elif breakout_up and vol_spike:
                desired_signal = SIZE
            
            # SHORT: CRSI > 80 (overbought) + volume spike + bearish 1d trend
            # OR breakout down with volume
            if crsi_val > 80 and vol_spike and not trend_bullish:
                desired_signal = -SIZE
            elif breakout_down and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: CRSI > 60 (mean revert) OR price breaks below channel
            if crsi_val > 60:
                exit_triggered = True
            if price_below_lower:
                exit_triggered = True
            # Also exit if RSI extreme opposite
            if rsi_val < 30:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: CRSI < 40 (mean revert) OR price breaks above channel
            if crsi_val < 40:
                exit_triggered = True
            if price_above_upper:
                exit_triggered = True
            # Also exit if RSI extreme opposite
            if rsi_val > 70:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Maintain position (no signal change = no fee)
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals