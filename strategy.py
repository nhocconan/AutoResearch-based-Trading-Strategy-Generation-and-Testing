#!/usr/bin/env python3
"""
Experiment #240: 1d Regime-Adaptive Strategy with Weekly HMA Trend + Choppiness Index
Hypothesis: Daily timeframe needs regime detection to switch between trend-following 
(breakout) and mean-reversion (RSI extremes) based on Choppiness Index. Weekly HMA 
provides macro trend bias to avoid counter-trend trades. This differs from previous 
attempts by using CHOP(14) to explicitly detect range vs trend regimes, then applying 
appropriate entry logic. Connors RSI for mean-reversion entries in range markets, 
Donchian breakout for trend markets. Position sizing: 0.25 entry, 0.125 half at 2R. 
Stoploss: 2.5*ATR trailing. Target: Beat Sharpe=0.499 with fewer but higher quality trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_chop_weekly_hma_crsi_donchian_atr_v1"
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
    Calculate Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values < 10 = oversold (long), > 90 = overbought (short)
    """
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Percent Rank of close over lookback
    percent_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = 100 * rank / (rank_period - 1)
    percent_rank[:rank_period] = 50.0
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return np.clip(crsi, 0, 100)

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    atr = calculate_atr(high, low, close, period)
    
    choppiness = np.zeros(len(close))
    for i in range(period, len(close)):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(atr[i-period+1:i+1])
        
        if highest_high - lowest_low > 0 and atr_sum > 0:
            choppiness[i] = 100 * np.log10((highest_high - lowest_low) / atr_sum) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    choppiness[:period] = 50.0
    return np.clip(choppiness, 0, 100)

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    choppiness = calculate_choppiness(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    sma_200 = calculate_sma(close, 200)
    
    # Track previous values for breakout detection
    prev_donchian_upper = np.roll(donchian_upper, 1)
    prev_donchian_lower = np.roll(donchian_lower, 1)
    prev_donchian_upper[0] = donchian_upper[0]
    prev_donchian_lower[0] = donchian_lower[0]
    
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
    
    for i in range(200, n):
        # Weekly trend filter (macro bias)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Regime detection via Choppiness Index
        is_range = choppiness[i] > 55.0  # Range/choppy market
        is_trend = choppiness[i] < 45.0  # Trending market
        
        # SMA 200 filter for long-term trend
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === MEAN REVERSION ENTRIES (Range Market) ===
        # Connors RSI extreme oversold + weekly bullish bias
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # === TREND FOLLOWING ENTRIES (Trend Market) ===
        # Donchian breakout
        breakout_long = close[i] > prev_donchian_upper[i]
        breakout_short = close[i] < prev_donchian_lower[i]
        
        # RSI pullback in trend
        rsi_pullback_long = 35 < rsi_14[i] < 50
        rsi_pullback_short = 50 < rsi_14[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Mean reversion: CRSI oversold in range market + weekly bullish
        if is_range and crsi_oversold and weekly_bullish:
            new_signal = SIZE_ENTRY
        
        # Mean reversion: CRSI oversold + above SMA200 (long-term uptrend)
        elif crsi_oversold and above_sma200 and weekly_bullish:
            new_signal = SIZE_ENTRY
        
        # Trend following: Donchian breakout + weekly bullish
        elif is_trend and breakout_long and weekly_bullish:
            new_signal = SIZE_ENTRY
        
        # Trend following: RSI pullback in uptrend + weekly bullish
        elif is_trend and rsi_pullback_long and weekly_bullish and above_sma200:
            new_signal = SIZE_ENTRY
        
        # Breakout with trend confirmation (always valid)
        elif breakout_long and weekly_bullish and above_sma200:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Mean reversion: CRSI overbought in range market + weekly bearish
        if is_range and crsi_overbought and weekly_bearish:
            new_signal = -SIZE_ENTRY
        
        # Mean reversion: CRSI overbought + below SMA200 (long-term downtrend)
        elif crsi_overbought and below_sma200 and weekly_bearish:
            new_signal = -SIZE_ENTRY
        
        # Trend following: Donchian breakout + weekly bearish
        elif is_trend and breakout_short and weekly_bearish:
            new_signal = -SIZE_ENTRY
        
        # Trend following: RSI pullback in downtrend + weekly bearish
        elif is_trend and rsi_pullback_short and weekly_bearish and below_sma200:
            new_signal = -SIZE_ENTRY
        
        # Breakout with trend confirmation (always valid)
        elif breakout_short and weekly_bearish and below_sma200:
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