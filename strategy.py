#!/usr/bin/env python3
"""
Experiment #148: 4h Regime-Adaptive Strategy with Daily HMA + Connors RSI
Hypothesis: 4h timeframe balances trend capture and trade frequency. Using
Choppiness Index to detect regime, then apply appropriate logic:
- Trending (CHOP<45): Follow Daily HMA direction with 4h pullback entries
- Ranging (CHOP>55): Mean-revert using Connors RSI extremes
Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
This adapts to both bull (2021) and bear/range (2022, 2025) markets.
Daily HMA provides major trend filter to avoid counter-trend trades.
Position sizing: 0.28 entry, 0.14 at 2R profit, stoploss at 2.5*ATR.
Timeframe: 4h for moderate trade frequency, lower fee drag than 15m/1h.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_crsi_daily_hma_adaptive_v1"
timeframe = "4h"
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
    CRSI < 10 = oversold, CRSI > 90 = overbought
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
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        pos_streak = max(0, streak[i])
        neg_streak = max(0, -streak[i])
        if pos_streak + neg_streak > 0:
            streak_rsi[i] = 100 * pos_streak / (pos_streak + neg_streak)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman's Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = 0
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=er_period, min_periods=er_period).sum().values
    volatility[:er_period] = 0
    
    er = np.zeros(n)
    for i in range(er_period, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing Constant
    sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1)) ** 2
    
    # KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period+1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness_index(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    hma_fast = calculate_hma(close, 10)
    hma_slow = calculate_hma(close, 30)
    
    # Bollinger Bands for squeeze detection
    bb_period = 20
    bb_sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_sma + 2.0 * bb_std
    bb_lower = bb_sma - 2.0 * bb_std
    bb_bw = (bb_upper - bb_lower) / bb_sma
    bb_position = (close - bb_lower) / (bb_upper - bb_lower)
    bb_position = np.nan_to_num(bb_position, nan=0.5)
    
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
    
    for i in range(150, n):
        # Daily trend filter (major trend direction)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 4h KAMA trend
        kama_trend_long = close[i] > kama[i]
        kama_trend_short = close[i] < kama[i]
        
        # HMA crossover
        hma_cross_long = hma_fast[i] > hma_slow[i]
        hma_cross_short = hma_fast[i] < hma_slow[i]
        
        # Regime detection via Choppiness Index
        trending_regime = chop[i] < 45.0
        ranging_regime = chop[i] > 55.0
        
        # CRSI extremes for mean reversion
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # RSI conditions
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Bollinger position
        bb_low = bb_position[i] < 0.15
        bb_high = bb_position[i] > 0.85
        
        # Volume confirmation (taker buy volume ratio)
        if 'taker_buy_volume' in prices.columns:
            taker_ratio = prices['taker_buy_volume'].values[i] / max(prices['volume'].values[i], 1)
            volume_bullish = taker_ratio > 0.55
            volume_bearish = taker_ratio < 0.45
        else:
            volume_bullish = True
            volume_bearish = True
        
        new_signal = 0.0
        
        # TRENDING REGIME: Follow trend with pullback entries
        if trending_regime:
            # LONG: Daily bullish + KAMA above + HMA cross + RSI pullback
            if daily_bullish and kama_trend_long and hma_cross_long:
                if rsi[i] < 55 or crsi[i] < 50:  # Pullback entry
                    new_signal = SIZE_ENTRY
            
            # SHORT: Daily bearish + KAMA below + HMA cross + RSI pullback
            elif daily_bearish and kama_trend_short and hma_cross_short:
                if rsi[i] > 45 or crsi[i] > 50:  # Pullback entry
                    new_signal = -SIZE_ENTRY
        
        # RANGING REGIME: Mean reversion with CRSI extremes
        elif ranging_regime:
            # LONG: CRSI oversold + BB lower + Daily not bearish
            if crsi_oversold and bb_low and not daily_bearish:
                new_signal = SIZE_ENTRY
            # SHORT: CRSI overbought + BB upper + Daily not bullish
            elif crsi_overbought and bb_high and not daily_bullish:
                new_signal = -SIZE_ENTRY
        
        # NEUTRAL REGIME: HMA/KAMA crossover with volume
        else:
            # LONG: KAMA cross up + volume bullish + Daily not bearish
            if kama_trend_long and hma_fast[i-1] <= hma_slow[i-1] and volume_bullish and not daily_bearish:
                new_signal = SIZE_ENTRY
            # SHORT: KAMA cross down + volume bearish + Daily not bullish
            elif kama_trend_short and hma_fast[i-1] >= hma_slow[i-1] and volume_bearish and not daily_bullish:
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