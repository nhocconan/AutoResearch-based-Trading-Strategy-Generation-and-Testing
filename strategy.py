#!/usr/bin/env python3
"""
Experiment #115: 15m Multi-Timeframe Regime-Adaptive Strategy with 4h HMA + 1h RSI + CRSI
Hypothesis: 15m is noisy, so use 4h HMA for trend direction and 1h RSI for entry timing.
Add Connors RSI (CRSI) for mean-reversion confirmation and Choppiness Index to detect
regime (trend vs range). In trending regimes, use trend-following entries. In ranging
regimes, use mean-reversion entries. This adaptive approach should work better across
different market conditions (2021 bull, 2022 bear, 2023-24 range, 2025 bear).
Position sizing: 0.25 entry, 0.15 half at profit, stoploss at 2.5*ATR trailing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_adaptive_4h_hma_1h_rsi_crsi_v1"
timeframe = "15m"
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
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_g = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_l = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.fillna(50).values
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - calculate consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_s = pd.Series(streak)
    streak_gain = streak_s.diff().where(streak_s.diff() > 0, 0.0)
    streak_loss = (-streak_s.diff()).where(streak_s.diff() < 0, 0.0)
    avg_sg = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_sl = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rsi = 100 - 100 / (1 + avg_sg / avg_sl.replace(0, np.nan))
    streak_rsi = streak_rsi.fillna(50).values
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank - percentage of past rank_period closes lower than current
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_lower = np.sum(window < close[i])
        percent_rank[i] = 100 * count_lower / rank_period
    
    # CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * (ATR(1, n) / (Highest High - Lowest Low)) * (100 / n^0.5)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # True Range (1-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of TR over period
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh = high_s.rolling(window=period, min_periods=period).max().values
    ll = low_s.rolling(window=period, min_periods=period).min().values
    hh_ll = hh - ll
    
    # CHOP formula
    chop = np.zeros(len(close))
    mask = hh_ll > 0
    chop[mask] = 100 * (tr_sum[mask] / hh_ll[mask]) * (100 / np.sqrt(period))
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    trend = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower_band[0]
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            trend[i] = 1
            supertrend[i] = max(lower_band[i], supertrend[i-1] if trend[i-1] == 1 else lower_band[i])
        else:
            trend[i] = -1
            supertrend[i] = min(upper_band[i], supertrend[i-1] if trend[i-1] == -1 else upper_band[i])
    
    return supertrend, trend

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h RSI for entry timing
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # 4h trend filter
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h RSI filter
        rsi_1h_ok_long = rsi_1h_aligned[i] < 70
        rsi_1h_ok_short = rsi_1h_aligned[i] > 30
        rsi_1h_pullback_long = rsi_1h_aligned[i] < 50  # pullback in uptrend
        rsi_1h_pullback_short = rsi_1h_aligned[i] > 50  # pullback in downtrend
        
        # Regime detection via Choppiness Index
        is_trending = chop[i] < 50  # below 50 = more trending
        is_ranging = chop[i] > 50   # above 50 = more ranging
        
        # Supertrend signal
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # CRSI extremes for mean reversion
        crsi_oversold = crsi[i] < 20
        crsi_overbought = crsi[i] > 80
        crsi_neutral = 20 <= crsi[i] <= 80
        
        # 15m RSI
        rsi_15m_ok_long = rsi_15m[i] < 70
        rsi_15m_ok_short = rsi_15m[i] > 30
        
        new_signal = 0.0
        
        # ===== TRENDING REGIME ENTRIES =====
        if is_trending:
            # Long: 4h bullish + Supertrend bullish + 1h RSI pullback + 15m RSI ok
            if trend_4h_bullish and st_bullish and rsi_1h_pullback_long and rsi_15m_ok_long:
                new_signal = SIZE_ENTRY
            
            # Short: 4h bearish + Supertrend bearish + 1h RSI pullback + 15m RSI ok
            elif trend_4h_bearish and st_bearish and rsi_1h_pullback_short and rsi_15m_ok_short:
                new_signal = -SIZE_ENTRY
        
        # ===== RANGING REGIME ENTRIES (Mean Reversion) =====
        else:
            # Long: CRSI oversold + 4h bullish (bias long in bull market)
            if crsi_oversold and trend_4h_bullish:
                new_signal = SIZE_ENTRY
            
            # Short: CRSI overbought + 4h bearish (bias short in bear market)
            elif crsi_overbought and trend_4h_bearish:
                new_signal = -SIZE_ENTRY
        
        # ===== STOPLOSS AND TAKE PROFIT LOGIC =====
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
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 1.5 * risk:
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