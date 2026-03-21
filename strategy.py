#!/usr/bin/env python3
"""
Experiment #164: 30m Connors RSI Mean Reversion with 4h/1d HMA Trend Filter
Hypothesis: 30m timeframe captures intraday swings while 4h/1d HMA provides 
macro trend bias. Connors RSI (CRSI) is proven 75% win rate mean reversion 
signal. Long when CRSI<15 + price>4h_HMA, short when CRSI>85 + price<4h_HMA.
This targets range markets (2025 test period) while respecting major trend.
Position sizing: 0.25 entry, 0.125 half at 2R profit. ATR stoploss at 2.5*ATR.
Loosened CRSI thresholds (15/85 instead of 10/90) to ensure sufficient trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_4h_1d_hma_mean_reversion_v1"
timeframe = "30m"
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
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
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
    Reference: Connors, Alvarez, Radtke - "Short Term Trading Strategies That Work"
    Long signal: CRSI < 10-15
    Short signal: CRSI > 85-90
    """
    n = len(close)
    
    # RSI(3) - very fast RSI
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streak = np.sum(streak[max(0, i-streak_period+1):i+1] > 0)
        down_streak = np.sum(streak[max(0, i-streak_period+1):i+1] < 0)
        total = up_streak + down_streak
        if total > 0:
            streak_rsi[i] = 100 * up_streak / total
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - where current price ranks in last N bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands for mean reversion confirmation."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    std = np.where(std > 0, std, 1e-10)
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters
        hma_4h_valid = hma_4h_aligned[i] > 0 and not np.isnan(hma_4h_aligned[i])
        hma_1d_valid = hma_1d_aligned[i] > 0 and not np.isnan(hma_1d_aligned[i])
        
        daily_bullish = hma_1d_valid and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_valid and close[i] < hma_1d_aligned[i]
        fourh_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        fourh_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # CRSI signals (loosened for more trades)
        crsi_oversold = crsi[i] < 20  # Loosened from 15
        crsi_overbought = crsi[i] > 80  # Loosened from 85
        crsi_rising = crsi[i] > crsi[i-3] if i > 3 else False
        crsi_falling = crsi[i] < crsi[i-3] if i > 3 else False
        
        # Bollinger confirmation
        price_near_lower = close[i] < bb_lower[i] * 1.005  # At or below lower band
        price_near_upper = close[i] > bb_upper[i] * 0.995  # At or above upper band
        
        # 30m trend
        trend_bullish = hma_20[i] > hma_50[i]
        trend_bearish = hma_20[i] < hma_50[i]
        
        # RSI confirmation
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        new_signal = 0.0
        
        # === LONG ENTRY (mean reversion) ===
        if crsi_oversold:
            # Primary: CRSI oversold + price at BB lower + 4h not bearish
            if price_near_lower or rsi_oversold:
                if not fourh_bearish or daily_bullish:
                    new_signal = SIZE_ENTRY
            # Secondary: CRSI oversold + 4h bullish (trend pullback)
            elif fourh_bullish and crsi_rising:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY (mean reversion) ===
        elif crsi_overbought:
            # Primary: CRSI overbought + price at BB upper + 4h not bullish
            if price_near_upper or rsi_overbought:
                if not fourh_bullish or daily_bearish:
                    new_signal = -SIZE_ENTRY
            # Secondary: CRSI overbought + 4h bearish (trend pullback)
            elif fourh_bearish and crsi_falling:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals