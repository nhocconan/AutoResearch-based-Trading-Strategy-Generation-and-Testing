#!/usr/bin/env python3
"""
Experiment #499: 1h Primary + 4h/12h HTF — Connors RSI + Choppiness + Session Filter

Hypothesis: 1h timeframe with strict confluence filters can achieve HTF-level trade frequency
with better entry timing. Using Connors RSI (proven 75% win rate) + Choppiness Index regime
filter + UTC session filter should generate 40-80 trades/year with high win rate.

Strategy logic:
1. 12h HMA(21) = primary trend bias (HTF filter)
2. 4h HMA(21) = secondary trend confirmation (faster HTF)
3. 1h Connors RSI = entry timing (CRSI<15 long, CRSI>85 short)
4. 1h Choppiness Index = regime filter (CHOP<38 trend, CHOP>62 range)
5. Session filter: 08-20 UTC only (high liquidity, avoid Asian chop)
6. ATR(14)*2.5 stoploss on all positions
7. Size: 0.20 base, 0.30 strong (when all 3 HTF align)

Key innovations:
- Connors RSI instead of standard RSI (faster mean reversion signal)
- Dual HTF (4h + 12h) for stronger trend confirmation
- Choppiness Index to switch between trend/mean-reversion logic
- Session filter to avoid low-liquidity periods (reduces false signals)

Target: Sharpe>0.40, trades=40-80/year, DD<-30%
Timeframe: 1h (with 4h/12h HTF for direction)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_session_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3) - fast RSI
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Streak RSI - based on consecutive up/down days
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
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        up_streaks = np.sum(streak[max(0,i-streak_period):i] > 0)
        down_streaks = np.sum(streak[max(0,i-streak_period):i] < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100.0 * up_streaks / total
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank - where current return ranks vs last 100 periods
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        returns = np.diff(close[max(0,i-rank_period):i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns)
            pct_rank[i] = 100.0 * rank
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_fast[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    CHOP > 61.8 = ranging (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest > lowest:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time_array // (1000 * 3600)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Extract UTC hour for session filter
    utc_hour = get_hour_from_open_time(open_time)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for primary trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h HMA for secondary trend confirmation
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        in_session = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        # === 12h HTF PRIMARY BIAS ===
        htf12_bull = close[i] > hma_12h_aligned[i]
        htf12_bear = close[i] < hma_12h_aligned[i]
        
        # === 4h HTF SECONDARY CONFIRMATION ===
        htf4_bull = close[i] > hma_4h_aligned[i]
        htf4_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF ALIGNMENT (both agree = strong signal) ===
        htf_aligned_bull = htf12_bull and htf4_bull
        htf_aligned_bear = htf12_bear and htf4_bear
        
        # === CHOPPINESS REGIME ===
        trending_regime = chop[i] < 38.2  # Trending market
        ranging_regime = chop[i] > 61.8   # Ranging market
        neutral_regime = not trending_regime and not ranging_regime
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === VOLATILITY FILTER ===
        atr_ratio = atr[i] / np.nanmean(atr[max(0,i-100):i]) if i >= 100 else 1.0
        vol_normal = atr_ratio < 2.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if in_session and vol_normal:
            # TREND LONG: HTF aligned bull + CRSI pullback in trending regime
            if htf_aligned_bull and trending_regime:
                if crsi_oversold and above_sma50:
                    desired_signal = SIZE_STRONG  # All confluence aligned
                elif crsi[i] < 30.0 and above_sma50:
                    desired_signal = SIZE_BASE  # Weaker pullback
            
            # TREND SHORT: HTF aligned bear + CRSI rally in trending regime
            elif htf_aligned_bear and trending_regime:
                if crsi_overbought and below_sma50:
                    desired_signal = -SIZE_STRONG  # All confluence aligned
                elif crsi[i] > 70.0 and below_sma50:
                    desired_signal = -SIZE_BASE  # Weaker rally
            
            # MEAN REVERSION LONG: Ranging regime + extreme CRSI
            elif ranging_regime:
                if crsi_extreme_oversold and above_sma200:
                    desired_signal = SIZE_BASE  # Mean revert long at support
                elif crsi_oversold and above_sma50:
                    desired_signal = SIZE_BASE * 0.8
            
            # MEAN REVERSION SHORT: Ranging regime + extreme CRSI
            elif ranging_regime:
                if crsi_extreme_overbought and below_sma200:
                    desired_signal = -SIZE_BASE  # Mean revert short at resistance
                elif crsi_overbought and below_sma50:
                    desired_signal = -SIZE_BASE * 0.8
            
            # NEUTRAL REGIME: Only take extreme CRSI signals with SMA200 filter
            elif neutral_regime:
                if crsi_extreme_oversold and above_sma200:
                    desired_signal = SIZE_BASE * 0.8
                elif crsi_extreme_overbought and below_sma200:
                    desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update highest since entry for trailing
            highest_since_entry = max(highest_since_entry, high[i])
            # Check stoploss
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trail stop: move up as price rises
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            # Update lowest since entry for trailing
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Check stoploss
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trail stop: move down as price falls
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                # Set stoploss
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