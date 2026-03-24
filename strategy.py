#!/usr/bin/env python3
"""
Experiment #606: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Daily timeframe with Connors RSI (CRSI) provides superior mean-reversion
entries during range markets, while 1w HMA provides macro trend bias. Choppiness Index
determines regime: CHOP>61.8 = mean revert (CRSI extremes), CHOP<38.2 = trend follow (HMA).

Key differences from failed #602 (4h_crsi_chop_hma_kama):
1. 1d timeframe instead of 4h - fewer trades, less fee drag, proven for BTC/ETH
2. Connors RSI (3-period RSI + streak RSI + percent rank) - 75% win rate in research
3. Simpler regime logic - CHOP determines mean-revert vs trend-follow
4. 1w HMA for macro bias only (not dual 1d+1w)
5. Fewer filters = more trades (avoid 0-trade failure)

Strategy logic:
1. 1w HMA(21) = macro trend bias (price > HMA = bull bias, only long CRSI)
2. 1d Choppiness(14) = regime (CHOP>61.8 = range, CHOP<38.2 = trend)
3. 1d Connors RSI = entry timing (CRSI<15 long, CRSI>85 short)
4. 1d SMA(200) = trend filter (price>SMA for longs, price<SMA for shorts)
5. ATR(14)*2.5 stoploss on all positions

Entry logic:
- RANGE (CHOP>55): Long CRSI<15 + price>SMA200, Short CRSI>85 + price<SMA200
- TREND (CHOP<45): Long price>1w_HMA + pullback to HMA(21), Short opposite
- TRANSITION: Reduced size or flat

Target: Sharpe>0.40, trades>=20/year (80+ train, 10+ test)
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean-reversion signals
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    RSI(close,3): Standard RSI on 3-period
    RSI(streak,2): RSI on consecutive up/down days streak
    PercentRank(100): Percentile rank of today's return vs last 100 days
    
    Long: CRSI < 10-15 (oversold)
    Short: CRSI > 85-90 (overbought)
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: RSI(3) on close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.zeros(n)
    rsi_close[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / (avg_streak_loss[i] + 1e-10)
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: PercentRank(100) - percentile of today's return
    returns = np.zeros(n)
    returns[0] = 0.0
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
        else:
            returns[i] = 0.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            count_below = np.sum(valid < returns[i])
            percent_rank[i] = 100.0 * count_below / len(valid)
        else:
            percent_rank[i] = 50.0
    
    # Combine components
    for i in range(max(rsi_period, streak_period, rank_period), n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    hma_1d = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
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
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(hma_1d[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w macro) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound (mean revert)
        chop_trend = chop[i] < 45.0   # Trending (trend follow)
        is_range = chop_range
        is_trend = chop_trend
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === TREND FILTERS ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        price_above_hma21 = close[i] > hma_1d[i]
        price_below_hma21 = close[i] < hma_1d[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion with CRSI extremes + SMA200 filter
        if is_range:
            # Long: CRSI oversold + price above SMA200 (bullish mean reversion)
            if crsi_extreme_oversold and price_above_sma200:
                desired_signal = SIZE_STRONG
            elif crsi_oversold and price_above_sma200 and htf_bull:
                desired_signal = SIZE_BASE
            # Short: CRSI overbought + price below SMA200 (bearish mean reversion)
            elif crsi_extreme_overbought and price_below_sma200:
                desired_signal = -SIZE_STRONG
            elif crsi_overbought and price_below_sma200 and htf_bear:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME: Follow 1w HMA bias with 1d HMA pullback
        elif is_trend:
            # Long: 1w bull + pullback to 1d HMA
            if htf_bull and price_above_hma21 and close[i] < hma_1d[i] * 1.02:
                desired_signal = SIZE_BASE
            # Short: 1w bear + rally to 1d HMA
            elif htf_bear and price_below_hma21 and close[i] > hma_1d[i] * 0.98:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL/TRANSITION: Only take extreme CRSI with strong HTF bias
        else:
            if crsi_extreme_oversold and htf_bull:
                desired_signal = SIZE_BASE * 0.8
            elif crsi_extreme_overbought and htf_bear:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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