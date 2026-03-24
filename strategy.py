#!/usr/bin/env python3
"""
Experiment #923: 6h Primary + 1d/1w HTF — CRSI Mean Reversion + CHOP Regime Filter

Hypothesis: 6h timeframe is underexplored middle ground between 4h (too many trades) 
and 12h (too few). Connors RSI (CRSI) provides proven 75% win rate mean reversion 
signal. Choppiness Index (CHOP) detects regime to switch between mean-revert (range) 
and trend-follow (trending). 1d/1w HTF provides directional bias.

Key innovations:
1. CRSI(3,2,100) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Entry long: CRSI < 15 + price > SMA(200) + HTF bullish
   - Entry short: CRSI > 85 + price < SMA(200) + HTF bearish
2. CHOP(14) regime filter:
   - CHOP > 61.8 = range → use CRSI mean reversion
   - CHOP < 38.2 = trend → use HMA trend following
   - 38.2-61.8 = neutral → reduce size or stay flat
3. 1d HMA(21) + 1w HMA(21) for HTF bias confluence
4. Discrete sizing: 0.0, ±0.25, ±0.30 with 2.5x ATR trailing stop
5. LOOSE CRSI thresholds (15/85 not 10/90) to ensure >=30 trades/train

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_chop_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_crsi(close):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile of current close vs last 100 closes
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan, dtype=np.float64)
    for i in range(2, n):
        total = avg_streak_gain[i] + avg_streak_loss[i]
        if total > 1e-10:
            rsi_streak[i] = 100.0 * avg_streak_gain[i] / total
        else:
            rsi_streak[i] = 50.0
    
    # PercentRank(100)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(100, n):
        window = close[i-99:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = 100.0 * rank / 100.0
    
    # CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(100, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trend (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling high/low
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    chop = np.full(n, np.nan, dtype=np.float64)
    log_n = np.log10(period)
    
    for i in range(period, n):
        if highest[i] > lowest[i] and atr_sum[i] > 0:
            chop[i] = 100.0 * np.log10(atr_sum[i] / (highest[i] - lowest[i])) / log_n
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    hma_6h_16 = calculate_hma(close, period=16)
    hma_6h_48 = calculate_hma(close, period=48)
    sma_200 = calculate_sma(close, period=200)
    crsi = calculate_crsi(close)
    chop = calculate_chop(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
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
    
    for i in range(250, n):  # Need 250 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h_16[i]) or np.isnan(hma_6h_48[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d + 1w HMA confluence) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong HTF bias when both agree
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        htf_neutral = (htf_1d_bull and htf_1w_bear) or (htf_1d_bear and htf_1w_bull)
        
        # === REGIME DETECTION (CHOP) ===
        regime_range = chop[i] > 61.8  # Mean reversion regime
        regime_trend = chop[i] < 38.2  # Trend following regime
        regime_neutral = not regime_range and not regime_trend
        
        # === SMA(200) FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CRSI MEAN REVERSION SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # LOOSE threshold for more trades
        crsi_overbought = crsi[i] > 85.0
        
        # === HMA TREND SIGNALS ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_6h_16[i-1]) and not np.isnan(hma_6h_48[i-1]):
            hma_crossover_long = (hma_6h_16[i-1] <= hma_6h_48[i-1]) and (hma_6h_16[i] > hma_6h_48[i])
            hma_crossover_short = (hma_6h_16[i-1] >= hma_6h_48[i-1]) and (hma_6h_16[i] < hma_6h_48[i])
        
        hma_6h_bull = hma_6h_16[i] > hma_6h_48[i]
        hma_6h_bear = hma_6h_16[i] < hma_6h_48[i]
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # RANGE REGIME (CHOP > 61.8) - Mean Reversion with CRSI
        if regime_range:
            # Long: CRSI oversold + HTF not strongly bearish + price above SMA200
            if crsi_oversold and not htf_strong_bear and price_above_sma200:
                desired_signal = SIZE_BASE
            # Short: CRSI overbought + HTF not strongly bullish + price below SMA200
            elif crsi_overbought and not htf_strong_bull and price_below_sma200:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME (CHOP < 38.2) - Trend Following with HMA
        elif regime_trend:
            # Long: HMA crossover up + HTF bullish bias
            if hma_crossover_long and (htf_strong_bull or htf_1d_bull):
                desired_signal = SIZE_STRONG
            # Short: HMA crossover down + HTF bearish bias
            elif hma_crossover_short and (htf_strong_bear or htf_1d_bear):
                desired_signal = -SIZE_STRONG
            # Trend continuation (looser)
            elif hma_6h_bull and htf_strong_bull:
                desired_signal = SIZE_BASE
            elif hma_6h_bear and htf_strong_bear:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME - Only take strong HTF confluence signals
        elif regime_neutral:
            if htf_strong_bull and hma_6h_bull and crsi_oversold:
                desired_signal = SIZE_BASE * 0.5
            elif htf_strong_bear and hma_6h_bear and crsi_overbought:
                desired_signal = -SIZE_BASE * 0.5
        
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
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE * 0.5
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