#!/usr/bin/env python3
"""
Experiment #376: 30m Primary + 4h/1d HTF — Choppiness + cRSI + Session Filter

Hypothesis: 30m timeframe with strict HTF trend filter + Choppiness regime detection
+ Connors RSI mean reversion will generate 40-80 trades/year with positive Sharpe.

Key innovations:
1. cRSI (Connors RSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Proven 75% win rate for mean reversion in research literature
2. Choppiness Index regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend
3. 4h HMA for trend BIAS only (not entry trigger) - prevents counter-trend trades
4. Session filter: 08-20 UTC only (highest volume, lowest whipsaw)
5. VERY strict entry: cRSI < 15 OR > 85 (extreme only) + HTF alignment + regime match

Position sizing: 0.20 base, 0.25 when 1d HTF also aligned
Stoploss: 2.5x ATR(14) from entry

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=5 test, ALL symbols positive
Trade frequency: ~60/year on 30m (2-3 per week per symbol)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_session_4h1d_v1"
timeframe = "30m"
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
    Connors RSI - combines 3 components for mean reversion signal
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Research shows 75% win rate when CRSI < 10 (long) or > 90 (short)
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        up_streak = max(0, streak[i])
        down_streak = max(0, -streak[i])
        
        # Simple streak RSI calculation
        if up_streak + down_streak > 0:
            streak_rsi[i] = 100.0 * up_streak / (up_streak + down_streak + 0.01)
        else:
            streak_rsi[i] = 50.0
    
    # Component 3: Percentile Rank of close over last 100 bars
    pr = np.zeros(n)
    pr[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i])
        pr[i] = 100.0 * rank / rank_period
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + pr[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range (mean reversion favorable)
    CHOP < 38.2 = trending (trend following favorable)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    hma_30m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_30m[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        # Extract hour from open_time (Unix timestamp in milliseconds)
        open_time_ms = prices["open_time"].iloc[i]
        hour_utc = (open_time_ms // 3600000) % 24
        
        in_session = 8 <= hour_utc <= 20
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = choppy/range (mean reversion)
        # CHOP < 38.2 = trending (breakout)
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === HTF TREND BIAS (4h and 1d) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 30m HMA TREND ===
        hma_bull = close[i] > hma_30m[i]
        hma_bear = close[i] < hma_30m[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === cRSI EXTREMES (VERY STRICT for few trades) ===
        # Research: cRSI < 15 = extreme oversold, cRSI > 85 = extreme overbought
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Must be in session (08-20 UTC)
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # REGIME: CHOPPY (mean reversion with cRSI)
        if is_choppy:
            # Long: cRSI extreme oversold + 4h HTF bull + above SMA200
            if crsi_oversold and htf_4h_bull and above_sma200:
                # Stronger size if 1d also bull
                desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
            
            # Short: cRSI extreme overbought + 4h HTF bear + below SMA200
            elif crsi_overbought and htf_4h_bear and below_sma200:
                # Stronger size if 1d also bear
                desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
        
        # REGIME: TRENDING (pullback to HMA with cRSI confirmation)
        elif is_trending:
            # Long: 4h bull + pullback to HMA (close < HMA but cRSI rising from oversold)
            if htf_4h_bull and hma_bear and crsi[i] < 40.0:
                desired_signal = SIZE_BASE
            
            # Short: 4h bear + rally to HMA (close > HMA but cRSI falling from overbought)
            elif htf_4h_bear and hma_bull and crsi[i] > 60.0:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals