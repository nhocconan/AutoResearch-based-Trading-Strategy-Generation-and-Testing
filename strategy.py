#!/usr/bin/env python3
"""
Experiment #017: 12h Connors RSI Mean Reversion + 1d HMA Trend Filter
Hypothesis: Connors RSI (CRSI) has proven 75% win rate for mean reversion entries.
Combined with 1d HMA trend filter, this should work in both bull and bear markets.

Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - CRSI<10 = extreme oversold (long entry)
   - CRSI>90 = extreme overbought (short entry)
2. 1d HMA trend filter: only long when price>1d_HMA, only short when price<1d_HMA
3. Volume spike confirmation: volume > 1.8x 20-bar MA (filters false reversals)
4. Asymmetric sizing: 0.30 when trend-aligned, 0.20 when counter-trend
5. 12h timeframe: captures multi-day swings, fewer trades = less fee drag
6. Simple 2.5*ATR trailing stop (no complex position tracking)

Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
Target: 30-60 trades/year, Sharpe > 0.5 on all symbols.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_1d_hma_volume_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI of streak (consecutive up/down days).
    Streak = number of consecutive days price moved in same direction.
    """
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        up_streaks = np.sum(streak[i-period+1:i+1] > 0)
        total = period
        if total > 0:
            streak_rsi[i] = 100 * up_streaks / total
        else:
            streak_rsi[i] = 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank: percentage of past 'period' closes below current close.
    """
    n = len(close)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        pr[i] = 100 * count_below / (period - 1) if period > 1 else 50.0
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_TREND = 0.30  # When trend-aligned
    SIZE_COUNTER = 0.20  # When counter-trend (riskier)
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - determines regime
        bull_trend = close[i] > hma_1d_aligned[i]
        bear_trend = close[i] < hma_1d_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.8 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # CRSI extremes (Connors RSI mean reversion signals)
        crsi_oversold = crsi[i] < 10  # Extreme oversold
        crsi_overbought = crsi[i] > 90  # Extreme overbought
        
        # Additional filter: price vs 50 EMA (secondary trend)
        price_above_ema50 = close[i] > ema_50[i]
        price_below_ema50 = close[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (CRSI < 10 = extreme oversold) ===
        if crsi_oversold:
            # Best long: CRSI oversold + bull trend (1d HMA) + volume spike
            if bull_trend and volume_confirmed:
                new_signal = SIZE_TREND
            # Moderate long: CRSI oversold + bull trend (no volume confirmation)
            elif bull_trend:
                new_signal = SIZE_COUNTER
            # Counter-trend long: CRSI oversold + bear trend + price below EMA50 (deep oversold)
            elif bear_trend and price_below_ema50:
                new_signal = SIZE_COUNTER
        
        # === SHORT ENTRIES (CRSI > 90 = extreme overbought) ===
        elif crsi_overbought:
            # Best short: CRSI overbought + bear trend (1d HMA) + volume spike
            if bear_trend and volume_confirmed:
                new_signal = -SIZE_TREND
            # Moderate short: CRSI overbought + bear trend (no volume confirmation)
            elif bear_trend:
                new_signal = -SIZE_COUNTER
            # Counter-trend short: CRSI overbought + bull trend + price above EMA50 (deep overbought)
            elif bull_trend and price_above_ema50:
                new_signal = -SIZE_COUNTER
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
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
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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