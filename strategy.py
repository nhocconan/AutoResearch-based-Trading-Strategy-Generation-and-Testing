#!/usr/bin/env python3
"""
Experiment #756: 30m Primary + 4h/1d HTF — Choppiness + cRSI + Session Filter

Hypothesis: 30m timeframe with 4h/1d HTF bias provides optimal trade frequency
(40-80/year) while maintaining signal quality. Uses 3+ confluence filters:
1. 4h HMA(21) for intermediate trend direction
2. 1d HMA(21) for regime filter (bull/bear market)
3. 30m Choppiness Index(14) to detect range vs trend regimes
4. 30m Connors RSI for mean-reversion entries in range, trend-following in trend
5. Session filter (08-20 UTC) to avoid low-volume periods

Key innovations:
- Choppiness > 61.8 = range (use cRSI mean-reversion)
- Choppiness < 38.2 = trend (use HMA crossover + cRSI pullback)
- Session filter reduces false signals during Asian low-volume hours
- Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn
- ATR(14) 2.5x trailing stop for risk management

Entry conditions (LOOSE enough for ≥30 trades/train, ≥3/test):
- LONG: 4h HMA bull + 1d HMA bull + (CHOP>61.8 + cRSI<20) OR (CHOP<38.2 + cRSI<40 + HMA cross)
- SHORT: 4h HMA bear + 1d HMA bear + (CHOP>61.8 + cRSI>80) OR (CHOP<38.2 + cRSI>60 + HMA cross)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%, trades/year 40-80
Timeframe: 30m
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_hma_4h1d_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - combines RSI, streak RSI, and percent rank"""
    n = len(close)
    if n < rank_period + rsi_period + 1:
        return np.full(n, np.nan)
    
    # RSI component (short period)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Streak RSI (use absolute streak values)
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, streak_period)
    # Invert for negative streaks
    for i in range(len(streak)):
        if streak[i] < 0 and not np.isnan(streak_rsi[i]):
            streak_rsi[i] = 100.0 - streak_rsi[i]
    
    # Percent rank component
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            pct_rank[i] = 100.0 * np.sum(returns[:-1] <= current_return) / (len(returns) - 1)
    
    # Combine components
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market ranging vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr_sum += tr
            
            choppiness[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hour = (open_time // 1000 // 3600) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (1d HMA) ===
        regime_bull = close[i] > hma_1d_aligned[i]
        regime_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = choppiness[i] > 61.8  # Ranging market
        chop_trend = choppiness[i] < 38.2  # Trending market
        chop_neutral = not chop_range and not chop_trend
        
        # === 30m HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_16[i-1]) and not np.isnan(hma_48[i-1]):
            hma_crossover_long = (hma_16[i-1] <= hma_48[i-1]) and (hma_16[i] > hma_48[i])
            hma_crossover_short = (hma_16[i-1] >= hma_48[i-1]) and (hma_16[i] < hma_48[i])
        
        # === 30m HMA TREND ===
        hma_30m_bull = hma_16[i] > hma_48[i]
        hma_30m_bear = hma_16[i] < hma_48[i]
        
        # === cRSI CONDITIONS ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        crsi_moderate_oversold = crsi[i] < 40.0
        crsi_moderate_overbought = crsi[i] > 60.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_4h_bull and regime_bull and in_session:
            # Range market: mean reversion with cRSI
            if chop_range and crsi_oversold:
                if crsi_extreme_oversold:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Trend market: pullback entry
            elif chop_trend and crsi_moderate_oversold and hma_30m_bull:
                if hma_crossover_long or crsi_extreme_oversold:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Neutral: use HMA crossover confirmation
            elif chop_neutral and hma_crossover_long and crsi_moderate_oversold:
                desired_signal = SIZE_BASE
        
        # SHORT entries
        elif htf_4h_bear and regime_bear and in_session:
            # Range market: mean reversion with cRSI
            if chop_range and crsi_overbought:
                if crsi_extreme_overbought:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            # Trend market: pullback entry
            elif chop_trend and crsi_moderate_overbought and hma_30m_bear:
                if hma_crossover_short or crsi_extreme_overbought:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            # Neutral: use HMA crossover confirmation
            elif chop_neutral and hma_crossover_short and crsi_moderate_overbought:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals