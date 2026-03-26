#!/usr/bin/env python3
"""
Experiment #011: 6h CRSI Momentum + 1d Trend Bias

HYPOTHESIS: Conners RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
is a proven momentum oscillator that catches reversals better than plain RSI.
Combined with 1d SMA200 trend alignment, this captures mean-reversion setups
in the direction of the larger trend. CRSI < 10 historically has 75% win rate.

CRSI hasn't been tried on 6h in any of the 27 failed attempts. The key is
LOOSE CRSI thresholds (< 15 for longs, > 85 for shorts) + trend confirmation.

TIMEFRAME: 6h primary
HTF: 1d for SMA200 trend bias
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Standard RSI"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return (100 - (100 / (1 + rs))).values

def calculate_rsi_streak(close, period=2):
    """RSI Streak - consecutive up/down closes"""
    n = len(close)
    delta = pd.Series(close).diff()
    
    # Mark up/down
    is_up = (delta > 0).astype(float)
    is_down = (delta < 0).astype(float) * -1
    
    # Cumulative streak
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = max(0, streak[i-1] + 1)
        elif delta.iloc[i] < 0:
            streak[i] = min(0, streak[i-1] - 1)
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak
    streak_series = pd.Series(streak)
    return calculate_rsi(streak_series.values, period)

def calculate_percent_rank(close, period=100):
    """Percent Rank over rolling window"""
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(period, n):
        window = close[i-period:i]
        count_below = np.sum(window < close[i])
        result[i] = (count_below / period) * 100
    
    return result

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Conners RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi3 = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, rank_period)
    
    crsi = (rsi3 + rsi_streak + pct_rank) / 3.0
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend bias
    sma_1d_raw = calculate_sma(df_1d['close'].values, period=200)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate local 6h indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    warmup = 200  # Need 200 bars for SMA200 alignment
    
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
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        crsi_val = crsi[i]
        
        # === TREND BIAS (1d SMA200) ===
        bullish_trend = close[i] > sma_1d_aligned[i]
        bearish_trend = close[i] < sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # CRSI deeply oversold (< 15) + price above SMA200 + volume confirmation
            if crsi_val < 15 and bullish_trend:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # CRSI extremely overbought (> 85) + price below SMA200 + volume confirmation
            if crsi_val > 85 and bearish_trend:
                desired_signal = -SIZE
        
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
        
        # === EXIT: CRSI mean reversion ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: CRSI crosses above 60 (mean reversion)
            if crsi_val > 60:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: CRSI crosses below 40 (mean reversion)
            if crsi_val < 40:
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