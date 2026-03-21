#!/usr/bin/env python3
"""
Experiment #375: 1h CRSI Mean Reversion + 4h HMA Trend + ATR Stoploss
Hypothesis: Connors RSI (CRSI) is proven for mean-reversion in bear/range markets (75% win rate).
2025+ test period is bearish/ranging, so mean-reversion should outperform pure trend-following.
CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3. Entry when CRSI<20 (long) or >80 (short).
4h HMA(21) provides trend bias filter - only take longs when price>4h_HMA, shorts when price<4h_HMA.
1h SMA(200) additional filter for regime detection. ATR(14) stoploss at 2.5x for risk management.
Position size: 0.25 discrete levels to minimize fee churn while maintaining exposure.
Timeframe: 1h (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 50-100 trades total, positive Sharpe on ALL symbols.
Key insight: Mean-reversion works better in bear markets (2025+) while trend filter protects from counter-trend trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_hma_trend_mean_reversion_atr_v1"
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

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI Streak component for CRSI.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak_rsi = np.zeros(n)
    
    for i in range(period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(0, i-period), -1):
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        # Convert streak to RSI-like value (0-100)
        if up_streak > 0:
            streak_rsi[i] = 100 * up_streak / period
        elif down_streak > 0:
            streak_rsi[i] = 100 * (period - down_streak) / period
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi[:period] = 50.0
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component for CRSI.
    Measures where current return ranks among last N periods.
    """
    n = len(close)
    pct_rank = np.zeros(n)
    
    returns = np.diff(close, prepend=close[0]) / (np.roll(close, 1) + 1e-10)
    returns[0] = 0.0
    
    for i in range(period, n):
        window_returns = returns[i-period+1:i+1]
        current_return = returns[i]
        
        # Count how many returns in window are less than current
        rank = np.sum(window_returns < current_return)
        pct_rank[i] = 100 * rank / period
    
    pct_rank[:period] = 50.0
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + rsi_streak + pct_rank) / 3.0
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    sma_200 = calculate_sma(close, 200)
    
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
    
    for i in range(200, n):  # Start after 200 bars for SMA200
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(crsi[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h regime filter
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # CRSI mean-reversion signals (LOOSE thresholds for trade frequency)
        crsi_oversold = crsi[i] < 25  # Long entry zone
        crsi_overbought = crsi[i] > 75  # Short entry zone
        
        # CRSI extreme (stronger signal)
        crsi_extreme_long = crsi[i] < 15
        crsi_extreme_short = crsi[i] > 85
        
        # CRSI recovering from extreme
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (Mean Reversion + Trend Filter) ===
        # Primary: CRSI oversold + 4h bullish + above SMA200
        if crsi_oversold and trend_bullish and above_sma200:
            new_signal = SIZE_ENTRY
        # Secondary: CRSI extreme long (overrides trend filter for strong MR)
        elif crsi_extreme_long and crsi_rising:
            new_signal = SIZE_ENTRY
        # Tertiary: CRSI oversold + above SMA200 (4h neutral ok)
        elif crsi_oversold and above_sma200 and crsi_rising:
            new_signal = SIZE_ENTRY
        # Quaternary: CRSI moderately oversold + 4h bullish (ensures frequency)
        elif crsi[i] < 35 and trend_bullish and crsi_rising:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (Mean Reversion + Trend Filter) ===
        # Primary: CRSI overbought + 4h bearish + below SMA200
        if crsi_overbought and trend_bearish and below_sma200:
            new_signal = -SIZE_ENTRY
        # Secondary: CRSI extreme short (overrides trend filter for strong MR)
        elif crsi_extreme_short and crsi_falling:
            new_signal = -SIZE_ENTRY
        # Tertiary: CRSI overbought + below SMA200 (4h neutral ok)
        elif crsi_overbought and below_sma200 and crsi_falling:
            new_signal = -SIZE_ENTRY
        # Quaternary: CRSI moderately overbought + 4h bearish (ensures frequency)
        elif crsi[i] > 65 and trend_bearish and crsi_falling:
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