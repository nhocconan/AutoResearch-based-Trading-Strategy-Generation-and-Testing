#!/usr/bin/env python3
"""
Experiment #006: 1d Connors RSI + Choppiness Index Regime + 1w HMA Trend
Hypothesis: Daily timeframe reduces fee drag while capturing major swings.
Connors RSI (CRSI) provides high-probability mean reversion entries (75% win rate).
Choppiness Index filters regime: CHOP>61.8 = range (use MR), CHOP<38.2 = trend.
1w HMA provides primary trend bias (HTF filter via mtf_data helper).
Entry: CRSI<15 + price>1w_HMA (long), CRSI>85 + price<1w_HMA (short).
Exit: 2*ATR trailing stop or CRSI mean reversion (crosses 50).
Conservative sizing (0.25-0.30) with discrete levels to minimize fee churn.
Must work on BTC/ETH through 2022 crash and 2025 bear market.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_hma_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3) - fast RSI for short-term oversold/overbought
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak (period=2)
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percentile Rank - where current close ranks in last rank_period bars
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i]) / rank_period * 100
        crsi[i] = (rsi_fast[i] + streak_rsi[i] + rank) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        tr_range = highest_high - lowest_low
        
        if tr_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / tr_range) / np.log10(period)
    
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Need 250 bars for all indicators to warm up
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # 1w trend bias (HTF)
        hma_1w_bullish = close[i] > hma_1w_aligned[i]
        hma_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Choppiness regime
        chop_range = chop[i] > 61.8  # Ranging market - favor mean reversion
        chop_trend = chop[i] < 38.2  # Trending market - favor trend following
        
        # CRSI extremes for mean reversion
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_neutral = 40 < crsi[i] < 60  # Exit zone
        
        # Bollinger Band position
        bb_low = close[i] < bb_lower[i]
        bb_high = close[i] > bb_upper[i]
        
        # EMA trend
        ema_bullish = close[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # CRSI mean reversion + 1w HMA bullish + ranging regime
        if crsi_oversold and hma_1w_bullish and (chop_range or bb_low):
            new_signal = SIZE_ENTRY
        # CRSI mean reversion + EMA bullish (trend pullback)
        elif crsi_oversold and ema_bullish and close[i] > ema_200[i]:
            new_signal = SIZE_ENTRY
        # BB bounce + 1w HMA bullish
        elif bb_low and hma_1w_bullish and crsi[i] < 30:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # CRSI mean reversion + 1w HMA bearish + ranging regime
        if crsi_overbought and hma_1w_bearish and (chop_range or bb_high):
            new_signal = -SIZE_ENTRY
        # CRSI mean reversion + EMA bearish (trend pullback)
        elif crsi_overbought and ema_bearish and close[i] < ema_200[i]:
            new_signal = -SIZE_ENTRY
        # BB rejection + 1w HMA bearish
        elif bb_high and hma_1w_bearish and crsi[i] > 70:
            new_signal = -SIZE_ENTRY
        
        # === EXIT SIGNALS ===
        # CRSI mean reversion exit (crosses neutral zone)
        if position_side > 0 and crsi_neutral:
            new_signal = 0.0
        if position_side < 0 and crsi_neutral:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals