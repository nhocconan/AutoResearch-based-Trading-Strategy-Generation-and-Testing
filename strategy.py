#!/usr/bin/env python3
"""
Experiment #133: 15m Multi-Regime Strategy with 4h HMA Trend + CRSI + Choppiness
Hypothesis: 15m timeframe captures intraday moves but needs strong HTF filter to avoid
noise. Use 4h HMA for major trend direction. Choppiness Index detects ranging vs trending
regimes on 15m. In trending regime: enter on RSI pullbacks in 4h trend direction.
In ranging regime: mean-revert on Connors RSI (CRSI) extremes. CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
This adapts to both bull trends and bear/range markets. Position sizing: 0.25 entry,
0.12 at 2R profit, stoploss at 2.5*ATR trailing. 15m generates more trades than 1h/4h/1d.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_hma_crsi_chop_regime_v1"
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
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
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
    Long: CRSI < 10, Short: CRSI > 90
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        pos_streak = max(0, streak[i])
        neg_streak = max(0, -streak[i])
        if pos_streak + neg_streak > 0:
            streak_rsi[i] = 100 * pos_streak / (pos_streak + neg_streak)
        else:
            streak_rsi[i] = 50
    
    # Percent Rank of price change over lookback
    for i in range(rank_period, n):
        changes = close[i-rank_period+1:i+1]
        pct_change = (close[i] - changes[:-1]) / changes[:-1] * 100
        if len(pct_change) > 0:
            rank = np.sum(pct_change <= pct_change[-1]) / len(pct_change) * 100
            crsi[i] = (rsi_short[i] + streak_rsi[i] + rank) / 3
        else:
            crsi[i] = 50
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    for i in range(period, n):
        hh_ll = hh[i] - ll[i]
        if hh_ll > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / hh_ll) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper[0]
    for i in range(1, n):
        if close[i-1] <= supertrend[i-1]:
            supertrend[i] = min(upper[i], supertrend[i-1])
            if close[i] > supertrend[i]:
                trend[i] = 1
                supertrend[i] = lower[i]
        else:
            supertrend[i] = max(lower[i], supertrend[i-1])
            if close[i] < supertrend[i]:
                trend[i] = -1
                supertrend[i] = upper[i]
    
    return supertrend, trend

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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_3 = calculate_rsi(close, 3)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness_index(high, low, close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
    # HMA for trend
    hma_fast = calculate_hma(close, 9)
    hma_slow = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # 4h trend filter (major trend direction)
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m HMA trend
        hma_trend_long = hma_fast[i] > hma_slow[i]
        hma_trend_short = hma_fast[i] < hma_slow[i]
        
        # Supertrend
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # Regime detection via Choppiness Index
        trending_regime = chop[i] < 50.0  # Lenient for more trades
        ranging_regime = chop[i] > 50.0
        
        # RSI conditions
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # CRSI extremes for mean reversion
        crsi_oversold = crsi[i] < 20
        crsi_overbought = crsi[i] > 80
        
        new_signal = 0.0
        
        # TRENDING REGIME: Follow trend with pullback entries
        if trending_regime:
            # LONG: 4h bullish + 15m HMA bullish + Supertrend bullish + RSI pullback
            if trend_4h_bullish and hma_trend_long and st_bullish and rsi_oversold:
                new_signal = SIZE_ENTRY
            # SHORT: 4h bearish + 15m HMA bearish + Supertrend bearish + RSI rally
            elif trend_4h_bearish and hma_trend_short and st_bearish and rsi_overbought:
                new_signal = -SIZE_ENTRY
        
        # RANGING REGIME: Mean reversion with CRSI extremes
        elif ranging_regime:
            # LONG: CRSI oversold + not strongly bearish on 4h
            if crsi_oversold and not trend_4h_bearish:
                new_signal = SIZE_ENTRY
            # SHORT: CRSI overbought + not strongly bullish on 4h
            elif crsi_overbought and not trend_4h_bullish:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
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