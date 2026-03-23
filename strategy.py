#!/usr/bin/env python3
"""
Experiment #723: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime Switch

Hypothesis: After 482 failed strategies, the pattern is clear:
1. Simple trend-following fails in bear/range markets (2022 crash, 2025 bear)
2. Connors RSI has proven 75% win rate for mean reversion entries
3. Choppiness Index is the BEST meta-filter for detecting range vs trend
4. 1d timeframe naturally limits trades to 20-50/year (avoids fee drag)
5. 1w HMA provides ultra-long-term trend bias without whipsaw

Strategy logic:
- CHOP > 61.8 = range regime → use Connors RSI mean reversion
- CHOP < 38.2 = trend regime → use HMA trend following
- 1w HMA determines long/short bias (only trade with weekly trend)
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Entry: CRSI < 15 (long) or CRSI > 85 (short) + regime confirmation
- Exit: CRSI crosses middle (50) or 2.5x ATR stoploss
- Size: 0.25-0.30 discrete levels

Why this should beat Sharpe=0.612 baseline:
- Connors RSI catches reversals that regular RSI misses
- Choppiness filter prevents trend strategies in chop (major failure mode)
- 1w HMA bias prevents counter-trend trades in strong trends
- 1d timeframe = natural trade frequency control (no fee death)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (proven higher TF works best)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_hma_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: rank of today's return vs last 100 days
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI of Streak - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    streak_rsi = np.full(n, 50.0)
    
    for i in range(streak_period, n):
        if not np.isnan(streak_abs[i]):
            # Simple mapping: longer streak = more extreme
            streak_rsi[i] = 50 + streak_sign[i] * min(streak_abs[i] * 10, 50)
    
    # Percent Rank of daily returns
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percent_rank[i] = np.sum(valid < returns[i]) / len(valid) * 100
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh_ll = np.zeros(n)
    for i in range(period-1, n):
        hh = np.nanmax(high[i-period+1:i+1])
        ll = np.nanmin(low[i-period+1:i+1])
        hh_ll[i] = hh - ll
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100 * np.log10(atr_sum / (hh_ll + 1e-10)) / np.log10(period)
    
    chop = np.where(chop_raw > 0, chop_raw, np.nan)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for ultra-long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    HALF_SIZE = 0.14
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if atr_1d[i] <= 1e-10 or np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_1d[i] > 61.8  # Range market - use mean reversion
        is_trending = chop_1d[i] < 38.2  # Trend market - use trend following
        # Neutral zone: 38.2-61.8 - reduce position or stay flat
        
        # === ULTRA-LONG-TERM TREND BIAS (1w HMA) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY CONDITIONS ===
        long_signal = False
        
        # Path 1: Choppy regime + CRSI oversold + bullish weekly bias
        if is_choppy and crsi_1d[i] < 15 and trend_1w_bullish:
            long_signal = True
        
        # Path 2: Choppy regime + CRSI oversold + above SMA200
        if is_choppy and crsi_1d[i] < 15 and above_sma200:
            long_signal = True
        
        # Path 3: Trending regime + pullback to HMA + CRSI not overbought
        if is_trending and trend_1w_bullish and crsi_1d[i] < 40 and above_sma200:
            long_signal = True
        
        # Path 4: Deep CRSI oversold (< 10) regardless of regime (strong mean revert)
        if crsi_1d[i] < 10 and above_sma200:
            long_signal = True
        
        # Path 5: CRSI crossing up from oversold + bullish bias
        if i > 1 and not np.isnan(crsi_1d[i-1]):
            if crsi_1d[i-1] < 20 and crsi_1d[i] > crsi_1d[i-1] and trend_1w_bullish:
                long_signal = True
        
        if long_signal:
            desired_signal = current_size
        
        # === SHORT ENTRY CONDITIONS ===
        short_signal = False
        
        # Path 1: Choppy regime + CRSI overbought + bearish weekly bias
        if is_choppy and crsi_1d[i] > 85 and trend_1w_bearish:
            short_signal = True
        
        # Path 2: Choppy regime + CRSI overbought + below SMA200
        if is_choppy and crsi_1d[i] > 85 and below_sma200:
            short_signal = True
        
        # Path 3: Trending regime + rally to HMA + CRSI not oversold
        if is_trending and trend_1w_bearish and crsi_1d[i] > 60 and below_sma200:
            short_signal = True
        
        # Path 4: Deep CRSI overbought (> 90) regardless of regime (strong mean revert)
        if crsi_1d[i] > 90 and below_sma200:
            short_signal = True
        
        # Path 5: CRSI crossing down from overbought + bearish bias
        if i > 1 and not np.isnan(crsi_1d[i-1]):
            if crsi_1d[i-1] > 80 and crsi_1d[i] < crsi_1d[i-1] and trend_1w_bearish:
                short_signal = True
        
        if short_signal:
            desired_signal = -current_size
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, go with weekly trend
        if long_signal and short_signal:
            if trend_1w_bullish:
                desired_signal = current_size
            elif trend_1w_bearish:
                desired_signal = -current_size
            else:
                desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not extremely overbought and weekly still bullish
                if crsi_1d[i] < 75 and trend_1w_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not extremely oversold and weekly still bearish
                if crsi_1d[i] > 25 and trend_1w_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI extremely overbought or weekly trend reverses
            if crsi_1d[i] > 80 or trend_1w_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI extremely oversold or weekly trend reverses
            if crsi_1d[i] < 20 or trend_1w_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals