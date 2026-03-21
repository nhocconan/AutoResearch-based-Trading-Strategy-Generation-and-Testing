#!/usr/bin/env python3
"""
Experiment #390: 1d Connors RSI + Weekly HMA Trend + Choppiness Regime + ATR Stop
Hypothesis: Connors RSI (CRSI) is a proven mean-reversion indicator with ~75% win rate
on daily timeframes. Combined with weekly HMA for trend bias and Choppiness Index to
filter ranging markets, this should capture pullbacks in established trends. Daily
timeframe naturally reduces trade frequency and fee drag. Weekly HMA (via mtf_data)
provides robust trend filter without look-ahead. ATR(14) stoploss at 2.5x protects
capital. Position size 0.25-0.30 discrete to minimize churn. Target: Beat Sharpe=0.499.
Key insight: CRSI extremes ( <15 long, >85 short) + weekly trend filter = high-probability
entries with fewer whipsaws than pure trend-following on daily.
Timeframe: 1d (REQUIRED), HTF: 1w for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_weekly_hma_chop_regime_atr_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    Proven mean-reversion indicator with ~75% win rate on daily timeframe.
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # RSI on streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    avg_streak_g = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_l = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    rs_streak = np.where(avg_streak_l > 0, avg_streak_g / avg_streak_l, 100.0)
    rsi_streak = 100 - 100 / (1 + rs_streak)
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # PercentRank: percentage of closes in lookback that are below current close
    for i in range(rank_period, n):
        lookback = close[i-rank_period+1:i+1]
        count_below = np.sum(lookback[:-1] < close[i])  # exclude current
        percent_rank = count_below / (rank_period - 1) * 100
        crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank) / 3
    
    crsi[:rank_period] = 50.0  # neutral before enough data
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (use mean-reversion)
    CHOP < 38.2 = trending market (use trend-following)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        if highest_high - lowest_low > 0:
            chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop[:period] = 50.0
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

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
    sma_200 = calculate_sma(close, 200)
    
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
    
    for i in range(250, n):  # Start after 250 bars for SMA200 + CRSI warmup
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Choppiness regime
        is_ranging = chop[i] > 55  # Mean-reversion favorable
        is_trending = chop[i] <= 55  # Trend-following favorable
        
        # SMA200 filter for long-term trend
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # CRSI extremes for mean-reversion entries
        crsi_oversold = crsi[i] < 20  # Very oversold
        crsi_overbought = crsi[i] > 80  # Very overbought
        crsi_extreme_oversold = crsi[i] < 15  # Extremely oversold
        crsi_extreme_overbought = crsi[i] > 85  # Extremely overbought
        
        # CRSI momentum (rising from oversold / falling from overbought)
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (CRSI mean-reversion with trend filter) ===
        # Primary: CRSI extreme oversold + Weekly bullish + Above SMA200
        if crsi_extreme_oversold and weekly_bullish and above_sma200:
            new_signal = SIZE_ENTRY
        # Secondary: CRSI oversold + Weekly bullish + Ranging market (mean-reversion)
        elif crsi_oversold and weekly_bullish and is_ranging:
            new_signal = SIZE_ENTRY
        # Tertiary: CRSI oversold + Above SMA200 + CRSI rising (momentum confirmation)
        elif crsi_oversold and above_sma200 and crsi_rising:
            new_signal = SIZE_ENTRY
        # Quaternary: CRSI extreme oversold alone (ensures trade frequency)
        elif crsi_extreme_oversold and crsi_rising:
            new_signal = SIZE_ENTRY
        # Quintenary: CRSI oversold + Weekly bullish (loose filter)
        elif crsi_oversold and weekly_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (CRSI mean-reversion with trend filter) ===
        # Primary: CRSI extreme overbought + Weekly bearish + Below SMA200
        if crsi_extreme_overbought and weekly_bearish and below_sma200:
            new_signal = -SIZE_ENTRY
        # Secondary: CRSI overbought + Weekly bearish + Ranging market (mean-reversion)
        elif crsi_overbought and weekly_bearish and is_ranging:
            new_signal = -SIZE_ENTRY
        # Tertiary: CRSI overbought + Below SMA200 + CRSI falling (momentum confirmation)
        elif crsi_overbought and below_sma200 and crsi_falling:
            new_signal = -SIZE_ENTRY
        # Quaternary: CRSI extreme overbought alone (ensures trade frequency)
        elif crsi_extreme_overbought and crsi_falling:
            new_signal = -SIZE_ENTRY
        # Quintenary: CRSI overbought + Weekly bearish (loose filter)
        elif crsi_overbought and weekly_bearish:
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