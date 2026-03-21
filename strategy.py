#!/usr/bin/env python3
"""
Experiment #162: 1d Connors RSI Mean Reversion with Weekly HMA Trend Filter
Hypothesis: Daily timeframe with Connors RSI (CRSI) captures short-term mean reversion
while Weekly HMA provides macro trend bias. CRSI combines RSI(3) + Streak RSI(2) + 
PercentRank(100) for high-probability reversal signals (75% win rate in literature).
Choppiness Index filters regime - only mean-revert in ranging markets (CHOP>50).
This should work in 2025 bear/range market while capturing 2021 bull trends.
Entry: CRSI<15 (long) or CRSI>85 (short) + weekly trend alignment + CHOP>50.
Stoploss: 2.5*ATR trailing. Position sizing: 0.30 entry, 0.15 half-profit at 2R.
Target: Sharpe>0.5, trades>20 on train, trades>5 on test, DD<-30%.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_weekly_hma_chop_atr_v1"
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

def calculate_streak_rsi(close, period=2):
    """
    Calculate Streak RSI component of Connors RSI.
    Measures consecutive up/down days as RSI.
    Reference: Connors, Alvarez, Radtke - "Short Term Trading Strategies That Work"
    """
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    gain_streak = np.where(streak > 0, streak, 0.0)
    loss_streak = np.where(streak < 0, -streak, 0.0)
    
    avg_g = pd.Series(gain_streak).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss_streak).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    streak_rsi = 100 - 100 / (1 + rs)
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component of Connors RSI.
    Measures current price change vs historical changes over period.
    """
    n = len(close)
    percent_rank = np.zeros(n)
    
    # Calculate daily returns
    returns = np.diff(close, prepend=close[0])
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current_return = returns[i]
        
        # Count how many values in window are less than current
        count_lower = np.sum(window < current_return)
        percent_rank[i] = count_lower / period * 100
    
    # Fill initial values
    percent_rank[:period] = 50.0
    
    return percent_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Reference: Connors, Alvarez, Radtke - "Short Term Trading Strategies That Work"
    Long entry: CRSI < 10-15
    Short entry: CRSI > 85-90
    """
    rsi_3 = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion favored)
    CHOP < 38.2 = trending market (trend following favored)
    Reference: E.W. Dreiss
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl > 0, range_hl, 1e-10)
    
    # Sum ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    chop = 100 * np.log10(atr_sum / (range_hl * period))
    chop = np.where(np.isnan(chop), 50.0, chop)
    chop = np.clip(chop, 0, 100)
    
    return chop

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
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    
    # Calculate SMA200 for additional trend filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
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
    
    for i in range(250, n):  # Need 250 bars for SMA200 + CRSI warmup
        # Weekly trend filter
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Long-term trend filter
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # Regime detection - only mean revert in ranging markets
        is_ranging = chop[i] > 45.0  # Loosened from 61.8 for more trades
        
        # CRSI signals
        crsi_oversold = crsi[i] < 20  # Loosened from 15 for more trades
        crsi_overbought = crsi[i] > 80  # Loosened from 85 for more trades
        crsi_rising = crsi[i] > crsi[i-2] if i > 2 else False
        crsi_falling = crsi[i] < crsi[i-2] if i > 2 else False
        
        new_signal = 0.0
        
        # === MEAN REVERSION LONG ===
        if crsi_oversold and is_ranging:
            # Require weekly trend not bearish (avoid catching falling knife in strong downtrend)
            if weekly_bullish or above_sma200 or crsi_rising:
                new_signal = SIZE_ENTRY
        
        # === MEAN REVERSION SHORT ===
        elif crsi_overbought and is_ranging:
            # Require weekly trend not bullish
            if weekly_bearish or below_sma200 or crsi_falling:
                new_signal = -SIZE_ENTRY
        
        # === TREND CONTINUATION (when not ranging) ===
        if not is_ranging and new_signal == 0.0:
            # Long: weekly bullish + CRSI recovering from oversold
            if weekly_bullish and crsi[i] < 50 and crsi_rising:
                if crsi[i-1] < 30:  # Was recently oversold
                    new_signal = SIZE_ENTRY
            
            # Short: weekly bearish + CRSI falling from overbought
            elif weekly_bearish and crsi[i] > 50 and crsi_falling:
                if crsi[i-1] > 70:  # Was recently overbought
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