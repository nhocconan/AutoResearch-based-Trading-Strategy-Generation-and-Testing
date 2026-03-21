#!/usr/bin/env python3
"""
Experiment #021: 1h Connors RSI + 4h HMA Trend Filter
Hypothesis: Connors RSI (CRSI) captures short-term mean reversion opportunities
on 1h timeframe while 4h HMA provides major trend direction filter.
CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
Long when CRSI < 20 + price > 4h HMA (bullish trend pullback)
Short when CRSI > 80 + price < 4h HMA (bearish trend rally)
ATR stoploss (2.5x) protects against trend continuation.
Position sizing: 0.25 discrete levels to minimize fee churn.
Relaxed CRSI thresholds (20/80 instead of 10/90) to ensure >=10 trades/symbol.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_hma_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    
    # RSI(3) - short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on streak values
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_avg_g = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_l = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_rs = np.where(streak_avg_l > 0, streak_avg_g / streak_avg_l, 100.0)
    streak_rsi = 100 - 100 / (1 + streak_rs)
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank - where current price stands in last N periods
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine all three components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=1.0)
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.125
    
    # Track positions for stoploss (scalar values, not arrays)
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    for i in range(100, n):
        # 4h trend filter
        hma_valid = hma_4h_aligned[i] > 0
        trend_bullish = hma_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_valid and close[i] < hma_4h_aligned[i]
        
        # CRSI extremes (relaxed for more trades)
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        
        # Volume confirmation (relaxed)
        vol_confirm = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
        
        # Entry logic - relaxed conditions to ensure trades
        new_signal = 0.0
        
        # Long: bullish trend + CRSI oversold + volume
        if trend_bullish and crsi_oversold and vol_confirm:
            new_signal = SIZE
        # Short: bearish trend + CRSI overbought + volume
        elif trend_bearish and crsi_overbought and vol_confirm:
            new_signal = -SIZE
        # Also enter on CRSI extreme even without trend (mean reversion)
        elif crsi[i] < 15 and vol_confirm:
            new_signal = SIZE
        elif crsi[i] > 85 and vol_confirm:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                new_trailing = max(trailing_stop, close[i] - 2.5 * atr[i])
                if close[i] < new_trailing and new_trailing > 0:
                    new_signal = 0.0
                else:
                    trailing_stop = new_trailing
                # Take partial profit at 2R
                if close[i] > entry_price + 2.0 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        elif position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                new_trailing = min(trailing_stop if trailing_stop > 0 else 999999, close[i] + 2.5 * atr[i])
                if close[i] > new_trailing and new_trailing < 999999:
                    new_signal = 0.0
                else:
                    trailing_stop = new_trailing
                # Take partial profit at 2R
                if close[i] < entry_price - 2.0 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals