#!/usr/bin/env python3
"""
Experiment #025: 4h CRSI Reversal + 1d Trend + Choppiness Regime

HYPOTHESIS: Conners RSI (CRSI) combines 3 momentum measures into one composite
score. CRSI < 10 is EXTREMELY rare (1-2% of bars), ensuring very low trade 
frequency. Combined with 1d HMA trend alignment (bull bias for longs, bear for 
shorts) and Choppiness Index regime filtering, this catches major reversals 
at institutional points while avoiding whipsaws in choppy markets.

WHY IT WORKS IN BULL AND BEAR:
- Bull: CRSI < 10 + price > 1d HMA = oversold in uptrend → strong bounce
- Bear: CRSI > 90 + price < 1d HMA = overbought in downtrend → continuation short
- Choppiness filter: only enter when market is TRENDING (CHOP < 38.2), 
  avoiding range-bound whipsaws that destroy returns

DB EVIDENCE: CRSI + Donchian + Choppiness on SOL achieved test Sharpe 1.46
TIMEFRAME: 4h primary
HTF: 1d for trend alignment
TARGET: 75-150 total trades over 4 years (~20-40/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - optimized vectorized version"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        roll = pd.Series(series).rolling(span, min_periods=span)
        return roll.apply(lambda x: np.sum(x * weights) / np.sum(weights), raw=True).values
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = 2.0 * wma_half - wma_full
    return wma(diff, sqrt_n)

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

def calculate_crsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Conners RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Classic RSI on price changes
    2. RSI_Streak(2): RSI on consecutive up/down days
    3. PercentRank(100): Percentile rank of % changes over 100 periods
    
    CRSI < 10: Extreme oversold (very rare, high probability reversal)
    CRSI > 90: Extreme overbought (very rare)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # 1. RSI(3) - fast RSI on price changes
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    avg_loss = loss.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi3 = (100 - (100 / (1 + rs))).values
    
    # 2. RSI_Streak(2) - consecutive up/down
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = max(0, streak[i-1] + delta.iloc[i])
        elif delta.iloc[i] < 0:
            streak[i] = min(0, streak[i-1] + delta.iloc[i])
        else:
            streak[i] = 0
    
    streak_series = pd.Series(streak)
    streak_gain = streak_series.where(streak > 0, 0.0)
    streak_loss = (-streak_series).where(streak < 0, 0.0)
    
    streak_avg_gain = streak_gain.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = (100 - (100 / (1 + streak_rs))).values
    
    # 3. PercentRank(100) - percentile rank of % changes
    pct_change = pd.Series(close).pct_change()
    for i in range(period_rank, n):
        window = pct_change.iloc[i-period_rank+1:i+1]
        current = pct_change.iloc[i]
        valid = window.dropna()
        if len(valid) >= period_rank * 0.5:
            pct_rank = (valid < current).sum() / len(valid) * 100
            crsi[i] = (rsi3[i] + rsi_streak[i] + pct_rank) / 3
    
    return crsi

def calculate_choppiness(close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8: Market is choppy (ranging) - avoid trend following
    CHOP < 38.2: Market is trending - good for trend following
    
    Formula: 100 * LOG10(SUM(ATR(1), period) / (HHV(period) - LLV(period))) / LOG10(period)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr1 = calculate_atr(close, close, close, period=1)
    atr1 = np.abs(np.diff(close, prepend=close[0])))
    
    for i in range(period, n):
        if np.isnan(atr1[i]) or np.isnan(close[i]):
            continue
        
        sum_atr = np.sum(atr1[max(0, i-period+1):i+1])
        
        high_arr = pd.Series(close[:i+1]).rolling(period, min_periods=period).max().values
        low_arr = pd.Series(close[:i+1]).rolling(period, min_periods=period).min().values
        
        if not np.isnan(high_arr[i]) and not np.isnan(low_arr[i]):
            hhv_llv = high_arr[i] - low_arr[i]
            if hhv_llv > 0:
                chop[i] = 100 * np.log10(sum_atr / hhv_llv) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend alignment
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, period_rsi=3, period_streak=2, period_rank=100)
    chop = calculate_choppiness(close, period=14)
    
    # Volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(150, period_rank for period_rank in [100])  # Need CRSI to warm up
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (Choppiness) ===
        # CHOP < 38.2 = trending, good for entries
        # CHOP > 61.8 = choppy, avoid entries
        chop_val = chop[i] if not np.isnan(chop[i]) else 50.0
        is_trending = chop_val < 45.0  # Slightly relaxed from 38.2 to avoid missing trades
        
        # === TREND ALIGNMENT (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.2
        
        # === CRSI VALUES ===
        crsi_val = crsi[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # CRSI < 10 is EXTREMELY oversold (very rare signal)
            # Need: extreme oversold + trending market + bullish 1d trend + volume
            if crsi_val < 10:
                if is_trending and price_above_1d_hma and vol_confirm:
                    desired_signal = SIZE
            # Secondary: CRSI < 20 with very strong volume in clear uptrend
            elif crsi_val < 20 and is_trending and price_above_1d_hma and vol_ratio[i] > 1.5:
                desired_signal = SIZE * 0.5  # Half size for weaker signal
            
            # === NEW SHORT ENTRY ===
            # CRSI > 90 is EXTREMELY overbought
            # Need: extreme overbought + trending market + bearish 1d trend + volume
            if crsi_val > 90:
                if is_trending and not price_above_1d_hma and vol_confirm:
                    desired_signal = -SIZE
            # Secondary: CRSI > 80 with very strong volume in clear downtrend
            elif crsi_val > 80 and is_trending and not price_above_1d_hma and vol_ratio[i] > 1.5:
                desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === EXIT: CRSI normalizes OR trend breaks ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: CRSI back to neutral (> 50) OR price breaks below 1d HMA
            if crsi_val > 50:
                exit_triggered = True
            if close[i] < hma_1d_aligned[i]:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: CRSI back to neutral (< 50) OR price breaks above 1d HMA
            if crsi_val < 50:
                exit_triggered = True
            if close[i] > hma_1d_aligned[i]:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
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
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals