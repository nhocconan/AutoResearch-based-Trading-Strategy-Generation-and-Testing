#!/usr/bin/env python3
"""
Experiment #255: 1h Connors RSI Mean Reversion + Choppiness Regime + 4h HMA Trend Filter
Hypothesis: Regular RSI pullback strategies failed badly (Sharpe -2 to -13) in recent experiments.
Connors RSI (CRSI) is DIFFERENT - combines RSI(3) + RSI_Streak(2) + PercentRank(100) for more
robust mean-reversion signals with 75% win rate in literature. Choppiness Index detects regime:
CHOP > 61.8 = range (use mean-reversion), CHOP < 38.2 = trend (use trend-follow). 4h HMA provides
primary trend bias to avoid counter-trend mean-reversion. This differs from failed RSI strategies
by using CRSI instead of RSI(14), adding regime filter, and looser entry thresholds to ensure
trades. Position sizing: 0.25 entry, 0.15 half at 2R profit. Stoploss: 2.5*ATR trailing stop.
Target: Beat Sharpe=0.499 with positive Sharpe on ALL symbols (BTC, ETH, SOL).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h_hma_atr_v1"
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
    Literature shows 75% win rate for CRSI < 10 long, CRSI > 90 short.
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
    
    # Convert streak to RSI-like value
    abs_streak = np.abs(streak)
    streak_gain = np.where(streak > 0, abs_streak, 0.0)
    streak_loss = np.where(streak < 0, abs_streak, 0.0)
    
    avg_streak_g = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_l = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = np.where(avg_streak_l > 0, avg_streak_g / avg_streak_l, 100.0)
    streak_rsi = 100 - 100 / (1 + streak_rs)
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank - where current price sits in last N periods
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window[:-1] < window[-1])
        percent_rank[i] = count_below / (rank_period - 1) * 100 if rank_period > 1 else 50
    
    # Fill early values
    percent_rank[:rank_period] = 50
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    choppiness = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50
    
    choppiness[:period] = 50
    choppiness = np.clip(choppiness, 0, 100)
    
    return choppiness

def calculate_percent_rank(close, period=100):
    """Calculate percentile rank of current close vs last N periods."""
    n = len(close)
    pr = np.zeros(n)
    for i in range(period, n):
        window = close[i-period+1:i+1]
        count_below = np.sum(window[:-1] < window[-1])
        pr[i] = count_below / (period - 1) * 100 if period > 1 else 50
    pr[:period] = 50
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # HTF trend filter (4h HMA)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # Regime detection via Choppiness Index
        is_choppy = choppiness[i] > 55  # Range market - favor mean reversion
        is_trending = choppiness[i] < 45  # Trend market - favor trend following
        
        # CRSI extremes for mean reversion (looser thresholds to ensure trades)
        crsi_oversold = crsi[i] < 25  # Was < 10, loosened for more trades
        crsi_overbought = crsi[i] > 75  # Was > 90, loosened for more trades
        crsi_extreme = crsi_oversold or crsi_overbought
        
        # CRSI turning up/down from extreme
        prev_crsi = crsi[i-1] if i > 0 else crsi[i]
        crsi_turning_up = crsi_oversold and crsi[i] > prev_crsi
        crsi_turning_down = crsi_overbought and crsi[i] < prev_crsi
        
        # Price position vs 4h HMA for trend confirmation
        price_above_hma = close[i] > hma_4h_aligned[i] * 1.002  # Small buffer
        price_below_hma = close[i] < hma_4h_aligned[i] * 0.998
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Mean reversion in choppy market with bullish HTF trend
        if crsi_turning_up and is_choppy:
            if trend_bullish or price_above_hma:
                new_signal = SIZE_ENTRY
            elif not trend_bearish:  # Neutral trend OK in choppy market
                new_signal = SIZE_ENTRY * 0.7
        
        # Mean reversion extreme oversold (any regime)
        elif crsi_oversold and crsi[i] < 15:
            if trend_bullish:
                new_signal = SIZE_ENTRY
            elif not is_trending:  # OK in choppy/neutral
                new_signal = SIZE_ENTRY * 0.7
        
        # Trend pullback in trending market
        elif is_trending and trend_bullish:
            if crsi[i] < 40 and crsi[i] > prev_crsi:  # Pullback in uptrend
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Mean reversion in choppy market with bearish HTF trend
        if crsi_turning_down and is_choppy:
            if trend_bearish or price_below_hma:
                new_signal = -SIZE_ENTRY
            elif not trend_bullish:  # Neutral trend OK in choppy market
                new_signal = -SIZE_ENTRY * 0.7
        
        # Mean reversion extreme overbought (any regime)
        elif crsi_overbought and crsi[i] > 85:
            if trend_bearish:
                new_signal = -SIZE_ENTRY
            elif not is_trending:  # OK in choppy/neutral
                new_signal = -SIZE_ENTRY * 0.7
        
        # Trend pullback in trending market
        elif is_trending and trend_bearish:
            if crsi[i] > 60 and crsi[i] < prev_crsi:  # Pullback in downtrend
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