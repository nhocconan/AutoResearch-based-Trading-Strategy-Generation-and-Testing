#!/usr/bin/env python3
"""
Experiment #177: 1h Connors RSI Mean Reversion with 4h/12h HMA Trend Filter
Hypothesis: Connors RSI (CRSI) is a proven mean-reversion indicator with ~75% win rate.
Combined with 4h HMA for trend bias and 12h HMA for macro filter, this captures
pullbacks in established trends. CRSI<15 for longs, CRSI>85 for shorts provides
more trade opportunities than extreme RSI (20/80). Choppiness Index filters out
choppy markets where mean reversion fails. ATR stoploss at 2.5*ATR limits drawdown.
Position sizing: 0.25 entry, reduce to 0.125 at 2R profit. Discrete levels minimize fees.
This targets the 2025 bear/range market where pure trend-following fails.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_12h_hma_chop_atr_v1"
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
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Short-term momentum
    RSI_Streak(2): Duration of current up/down streak
    PercentRank(100): Percentile of today's return over last 100 periods
    
    Entry: CRSI < 10 (oversold) or CRSI > 90 (overbought)
    Reference: Connors & Alvarez, "Short Term Trading Strategies That Work"
    """
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak duration
    # Streak = consecutive days of positive/negative returns
    returns = np.diff(close, prepend=close[0])
    streak = np.zeros(n)
    for i in range(1, n):
        if returns[i] > 0:
            streak[i] = streak[i-1] + 1 if returns[i-1] > 0 else 1
        elif returns[i] < 0:
            streak[i] = streak[i-1] - 1 if returns[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    streak_abs = np.where(streak_abs == 0, 1, streak_abs)
    rsi_streak = calculate_rsi(streak_abs, streak_period)
    
    # Component 3: Percentile rank of returns over last 100 periods
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns[i-rank_period+1:i+1]
        current_return = returns[i]
        rank = np.sum(window_returns < current_return)
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    crsi[:rank_period] = 50.0  # Warm-up period
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion favorable)
    CHOP < 38.2 = trending market (trend following favorable)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl > 0, range_hl, 1e-10)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    chop = 100 * np.log10(atr_sum / (range_hl * period))
    chop = np.where(np.isnan(chop), 50.0, chop)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank."""
    n = len(series)
    pr = np.zeros(n)
    for i in range(window, n):
        window_vals = series[i-window+1:i+1]
        current = series[i]
        rank = np.sum(window_vals < current)
        pr[i] = 100.0 * rank / window
    pr[:window] = 50.0
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    
    # Calculate 1h HMA for additional trend confirmation
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
    
    for i in range(150, n):
        # HTF trend filters
        hma_4h_valid = hma_4h_aligned[i] > 0
        hma_12h_valid = hma_12h_aligned[i] > 0
        
        price_above_4h = close[i] > hma_4h_aligned[i] if hma_4h_valid else True
        price_below_4h = close[i] < hma_4h_aligned[i] if hma_4h_valid else True
        price_above_12h = close[i] > hma_12h_aligned[i] if hma_12h_valid else True
        price_below_12h = close[i] < hma_12h_aligned[i] if hma_12h_valid else True
        
        # 1h trend
        trend_bullish = hma_20[i] > hma_50[i]
        trend_bearish = hma_20[i] < hma_50[i]
        
        # Regime detection - mean reversion works better in ranging markets
        is_ranging = chop[i] > 50.0  # Loosened for more trades
        is_trending = chop[i] < 45.0
        
        # CRSI signals
        crsi_oversold = crsi[i] < 20  # Loosened from 10 for more trades
        crsi_overbought = crsi[i] > 80  # Loosened from 90 for more trades
        crsi_rising = crsi[i] > crsi[i-2] if i > 2 else False
        crsi_falling = crsi[i] < crsi[i-2] if i > 2 else False
        
        new_signal = 0.0
        
        # === MEAN REVERSION MODE (ranging market - CRSI excels here) ===
        if is_ranging:
            # Long: CRSI oversold + 4h trend not bearish
            if crsi_oversold and crsi_rising:
                if price_above_4h or (not price_below_4h and trend_bullish):
                    new_signal = SIZE_ENTRY
            
            # Short: CRSI overbought + 4h trend not bullish
            elif crsi_overbought and crsi_falling:
                if price_below_4h or (not price_above_4h and trend_bearish):
                    new_signal = -SIZE_ENTRY
        
        # === TREND PULLBACK MODE (trending market) ===
        elif is_trending:
            # Long pullback: uptrend + CRSI dip
            if trend_bullish and price_above_4h and price_above_12h:
                if crsi_oversold and crsi_rising:
                    new_signal = SIZE_ENTRY
            
            # Short pullback: downtrend + CRSI spike
            elif trend_bearish and price_below_4h and price_below_12h:
                if crsi_overbought and crsi_falling:
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