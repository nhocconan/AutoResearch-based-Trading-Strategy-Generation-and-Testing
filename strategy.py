#!/usr/bin/env python3
"""
Experiment #736: 30m Primary + 4h/1d HTF — Choppiness + cRSI + Session Filter

Hypothesis: 30m timeframe with 4h/1d HTF bias + Choppiness regime filter + Connors RSI
provides optimal edge for bear/range markets (2025 test period). 

Key innovations:
1. 4h HMA(21) for HTF trend bias — only trade with HTF direction
2. 30m Choppiness Index(14) — CHOP>61.8 = range (mean revert), CHOP<38.2 = trend
3. 30m Connors RSI (CRSI) — (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI<10, Short: CRSI>90 (proven 75% win rate in research)
4. Session filter: 08-20 UTC only (avoid low liquidity whipsaws)
5. Asymmetric sizing: 0.20 base, 0.30 when 3+ confluence
6. 2.5x ATR(14) trailing stop for risk management

Entry conditions (balanced for 40-80 trades/year):
- LONG: 4h HMA bull + CHOP>50 (range) + CRSI<15 + session 08-20 UTC
- SHORT: 4h HMA bear + CHOP>50 (range) + CRSI>85 + session 08-20 UTC
- TREND entries: CHOP<40 + HMA crossover + RSI(14)<40/>60

Target: Sharpe>0.40, trades>=40/year train, trades>=3 test, DD>-40%
Timeframe: 30m
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_session_4h1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    chop = np.zeros(n)
    chop[:] = np.nan
    for i in range(period, n):
        range_hl = hh[i] - ll[i]
        if range_hl > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / range_hl) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator
    Formula: (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(streak): RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    
    CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Calculate streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    
    if n >= streak_period + 1:
        for i in range(streak_period, n):
            # Calculate RSI on streak values (treating as price)
            delta_streak = np.diff(streak_abs[:i+1], prepend=streak_abs[0])
            gain = np.where(delta_streak > 0, delta_streak, 0.0)
            loss = np.where(delta_streak < 0, -delta_streak, 0.0)
            
            avg_gain = np.mean(gain[-streak_period:]) if len(gain) >= streak_period else 0
            avg_loss = np.mean(loss[-streak_period:]) if len(loss) >= streak_period else 0
            
            if avg_loss > 1e-10:
                rs = avg_gain / avg_loss
                rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
            else:
                rsi_streak[i] = 100.0
    
    # PercentRank(100) - percentile of today's return
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        window_returns = returns[i-rank_period+1:i+1]
        current_return = returns[i]
        rank = np.sum(window_returns <= current_return)
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hour = pd.to_datetime(open_time, unit='ms').hour
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
    rsi_14 = calculate_rsi(close, period=14)
    rsi_3 = calculate_rsi(close, period=3)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
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
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(rsi_14[i]):
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
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === HTF BIAS (4h + 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong HTF bias when 4h and 1d agree
        htf_strong_bull = htf_4h_bull and htf_1d_bull
        htf_strong_bear = htf_4h_bear and htf_1d_bear
        
        # === CHOPPINESS REGIME ===
        chop_range = chop_14[i] > 50.0  # Range/choppy market
        chop_trend = chop_14[i] < 45.0  # Trending market
        
        # === CRSI MEAN REVERSION ===
        crsi_oversold = crsi[i] < 15.0  # Long signal
        crsi_overbought = crsi[i] > 85.0  # Short signal
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === 30m HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_16[i-1]) and not np.isnan(hma_48[i-1]):
            hma_crossover_long = (hma_16[i-1] <= hma_48[i-1]) and (hma_16[i] > hma_48[i])
            hma_crossover_short = (hma_16[i-1] >= hma_48[i-1]) and (hma_16[i] < hma_48[i])
        
        # === 30m RSI CONDITIONS ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        desired_signal = 0.0
        confluence_count = 0
        
        # LONG ENTRIES
        if htf_4h_bull and in_session:
            confluence_count = 0
            
            # Mean reversion in range (primary signal)
            if chop_range and crsi_oversold:
                confluence_count += 2  # CHOP + CRSI
                if htf_strong_bull:
                    confluence_count += 1
                if rsi_oversold:
                    confluence_count += 1
                if crsi_extreme_oversold:
                    confluence_count += 1
            
            # Trend continuation
            elif chop_trend and hma_crossover_long:
                confluence_count += 2  # CHOP + HMA cross
                if rsi_oversold:
                    confluence_count += 1
                if htf_strong_bull:
                    confluence_count += 1
            
            # Set signal based on confluence
            if confluence_count >= 3:
                if crsi_extreme_oversold or hma_crossover_long:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT ENTRIES
        elif htf_4h_bear and in_session:
            confluence_count = 0
            
            # Mean reversion in range (primary signal)
            if chop_range and crsi_overbought:
                confluence_count += 2  # CHOP + CRSI
                if htf_strong_bear:
                    confluence_count += 1
                if rsi_overbought:
                    confluence_count += 1
                if crsi_extreme_overbought:
                    confluence_count += 1
            
            # Trend continuation
            elif chop_trend and hma_crossover_short:
                confluence_count += 2  # CHOP + HMA cross
                if rsi_overbought:
                    confluence_count += 1
                if htf_strong_bear:
                    confluence_count += 1
            
            # Set signal based on confluence
            if confluence_count >= 3:
                if crsi_extreme_overbought or hma_crossover_short:
                    desired_signal = -SIZE_STRONG
                else:
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