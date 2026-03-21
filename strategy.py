#!/usr/bin/env python3
"""
Experiment #198: 1d Connors RSI Mean Reversion with 4h HMA Trend Filter
Hypothesis: Daily timeframe is ideal for Connors RSI mean reversion strategy.
CRSI combines RSI(3) + RSI_Streak(2) + PercentRank(100) for precise entry timing.
4h HMA provides trend bias (long only when price > 4h HMA, short when <).
Bollinger Band Width detects regime: narrow bands = mean reversion opportunity.
This should generate 20-40 trades/year on 1d, enough for statistical significance.
Position sizing: 0.25 entry, stoploss at 2.5*ATR. Target: Beat Sharpe=0.499.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_4h_hma_bb_regime_atr_v1"
timeframe = "1d"
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
    """
    n = len(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percent Rank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        lookback = close[i-rank_period:i]
        count_below = np.sum(lookback < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    band_width = (upper - lower) / sma * 100.0
    return upper, lower, sma, band_width

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
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    
    # Calculate BB Width percentile for regime detection
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=100).apply(
        lambda x: np.sum(x < x[-1]) / len(x) * 100, raw=True
    ).values
    bb_width_percentile = np.nan_to_num(bb_width_percentile, 50.0)
    
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
        # HTF trend filter (4h HMA)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # Regime detection (BB Width percentile)
        regime_mean_revert = bb_width_percentile[i] < 40  # Narrow bands
        regime_trend = bb_width_percentile[i] > 60  # Wide bands
        
        # CRSI extreme levels for mean reversion
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_neutral = 25 < crsi[i] < 75
        
        # Price position relative to BB
        price_near_lower = close[i] < bb_lower[i] * 1.01
        price_near_upper = close[i] > bb_upper[i] * 0.99
        price_above_mid = close[i] > bb_mid[i]
        price_below_mid = close[i] < bb_mid[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Mean reversion: CRSI oversold + price near lower BB + bullish trend
        if crsi_oversold and price_near_lower:
            if trend_bullish:
                new_signal = SIZE_ENTRY
            elif regime_mean_revert:
                new_signal = SIZE_ENTRY * 0.8
        
        # Trend continuation: CRSI neutral + price above mid + bullish trend
        elif crsi_neutral and price_above_mid and trend_bullish:
            # Enter on pullback to mid
            if i > 1 and close[i-1] < bb_mid[i-1] and close[i] > bb_mid[i]:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Mean reversion: CRSI overbought + price near upper BB + bearish trend
        if crsi_overbought and price_near_upper:
            if trend_bearish:
                new_signal = -SIZE_ENTRY
            elif regime_mean_revert:
                new_signal = -SIZE_ENTRY * 0.8
        
        # Trend continuation: CRSI neutral + price below mid + bearish trend
        elif crsi_neutral and price_below_mid and trend_bearish:
            # Enter on pullback to mid
            if i > 1 and close[i-1] > bb_mid[i-1] and close[i] < bb_mid[i]:
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