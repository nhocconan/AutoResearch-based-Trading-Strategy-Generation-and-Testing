#!/usr/bin/env python3
"""
Experiment #333: 1h CRSI Mean Reversion + Choppiness Regime + 4h HMA Trend
Hypothesis: 1h strategies failed because they used single-regime logic. This strategy
adapts to market conditions: CRSI mean reversion in choppy markets (CHOP>61.8),
trend pullback in trending markets (CHOP<38.2). 4h HMA provides macro bias to avoid
counter-trend mean reversion. CRSI combines RSI(3)+RSI_Streak+PercentRank for faster
signals than standard RSI(14). Target: Beat Sharpe=0.499 with regime-adaptive logic.
Timeframe: 1h (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Key insight: Regime detection prevents mean reversion in strong trends (major failure mode).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h_hma_adaptive_atr_v1"
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
    Better for mean reversion than standard RSI(14).
    """
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak component (consecutive up/down days)
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like scale (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(len(close))
    for i in range(streak_period, len(close)):
        up_streaks = np.sum(streak[max(0, i-streak_period+1):i+1] > 0)
        streak_rsi[i] = (up_streaks / streak_period) * 100 if streak_period > 0 else 50
    
    # Percent Rank component (where does current price rank in last 100 bars?)
    percent_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_val) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

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
    chop = calculate_choppiness(high, low, close, 14)
    
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
    
    for i in range(250, n):  # Start after 250 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        daily_bullish = not np.isnan(hma_4h_aligned[i]) and close[i] > hma_4h_aligned[i]
        daily_bearish = not np.isnan(hma_4h_aligned[i]) and close[i] < hma_4h_aligned[i]
        
        # Regime detection
        is_choppy = chop[i] > 61.8  # Range market - use mean reversion
        is_trending = chop[i] < 38.2  # Trend market - use trend follow
        
        # CRSI extreme levels for mean reversion
        crsi_oversold = crsi[i] < 15  # Very oversold
        crsi_overbought = crsi[i] > 85  # Very overbought
        crsi_moderate_oversold = crsi[i] < 25
        crsi_moderate_overbought = crsi[i] > 75
        
        new_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion) ===
        if is_choppy:
            # Long: CRSI very oversold + 4h trend not strongly bearish
            if crsi_oversold and not daily_bearish:
                new_signal = SIZE_ENTRY
            # Short: CRSI very overbought + 4h trend not strongly bullish
            elif crsi_overbought and not daily_bullish:
                new_signal = -SIZE_ENTRY
            # Moderate entries with stronger 4h confirmation
            elif crsi_moderate_oversold and daily_bullish:
                new_signal = SIZE_ENTRY
            elif crsi_moderate_overbought and daily_bearish:
                new_signal = -SIZE_ENTRY
        
        # === REGIME 2: TRENDING MARKET (Trend Following) ===
        elif is_trending:
            # Long pullback: 4h bullish + CRSI recovering from oversold
            if daily_bullish and crsi[i] > 30 and crsi[i] < 50:
                new_signal = SIZE_ENTRY
            # Short pullback: 4h bearish + CRSI falling from overbought
            elif daily_bearish and crsi[i] < 70 and crsi[i] > 50:
                new_signal = -SIZE_ENTRY
            # Strong trend continuation
            elif daily_bullish and crsi[i] > 50:
                new_signal = SIZE_ENTRY
            elif daily_bearish and crsi[i] < 50:
                new_signal = -SIZE_ENTRY
        
        # === REGIME 3: NEUTRAL (No strong signal) ===
        else:
            # Only take high-probability setups
            if crsi_oversold and daily_bullish:
                new_signal = SIZE_ENTRY
            elif crsi_overbought and daily_bearish:
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