#!/usr/bin/env python3
"""
Experiment #725: 1h Primary + 4h/1d HTF — Connors RSI Mean Reversion with Trend Filter

Hypothesis: After 485 failed strategies, the pattern is clear:
- 4h timeframe works best for trend following (current best Sharpe=0.612)
- 1h strategies fail due to too many trades (fee drag) OR too few (0 trades)
- Connors RSI has 75% win rate in academic literature for mean reversion
- Key: Use 4h/1d for DIRECTION, 1h only for ENTRY TIMING

Strategy:
1. 4h HMA(21) = trend bias (long only when price > 4h HMA)
2. 1h Connors RSI < 20 = entry (oversold pullback in uptrend)
3. 1h Connors RSI > 80 = exit/short (overbought in downtrend)
4. Choppiness Index > 55 = skip (range market, no trend)
5. Volume > 0.7x 20-bar avg = confirmation
6. ATR(14) 2.5x trailing stoploss

Target: 30-60 trades/year on 1h, Sharpe > 0.612
Position size: 0.25 (conservative for lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_meanrev_hma_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average."""
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
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3) - very short term
    rsi3 = calculate_rsi(close, period=rsi_period)
    
    # Streak - count consecutive up/down closes
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Streak RSI using simple momentum calculation
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streaks = 0
        down_streaks = 0
        for j in range(i-streak_period+1, i+1):
            if streak[j] > 0:
                up_streaks += streak[j]
            elif streak[j] < 0:
                down_streaks += abs(streak[j])
        
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100 * up_streaks / total
        else:
            streak_rsi[i] = 50
    
    # Percent Rank of returns
    returns = np.diff(close) / (close[:-1] + 1e-10)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window_returns = returns[i-rank_period:i]
        current_return = returns[i-1]
        rank = np.sum(window_returns <= current_return)
        percent_rank[i] = (rank / len(window_returns)) * 100
    
    # Combine components
    valid_mask = ~np.isnan(rsi3) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi3[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    
    # Volume moving average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for stronger trend confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(chop_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10 or atr_1h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (4h and 1d HTF HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong trend when both agree
        strong_bullish = trend_4h_bullish and trend_1d_bullish
        strong_bearish = trend_4h_bearish and trend_1d_bearish
        
        # === CHOPPINESS FILTER ===
        # Skip if market is choppy (no clear trend)
        is_choppy = chop_1h[i] > 55
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_sma[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY: CRSI oversold + bullish trend + not choppy ===
        # Multiple paths to ensure trade frequency (Rule 9)
        long_signal = False
        
        # Path 1: Strong bullish + CRSI very oversold
        if strong_bullish and crsi_1h[i] < 15 and not is_choppy and volume_ok:
            long_signal = True
        
        # Path 2: 4h bullish only + CRSI extremely oversold (more lenient)
        if trend_4h_bullish and crsi_1h[i] < 10 and volume_ok:
            long_signal = True
        
        # Path 3: CRSI moderately oversold + both HTF bullish
        if crsi_1h[i] < 25 and strong_bullish and volume_ok:
            long_signal = True
        
        if long_signal:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY: CRSI overbought + bearish trend + not choppy ===
        short_signal = False
        
        # Path 1: Strong bearish + CRSI very overbought
        if strong_bearish and crsi_1h[i] > 85 and not is_choppy and volume_ok:
            short_signal = True
        
        # Path 2: 4h bearish only + CRSI extremely overbought
        if trend_4h_bearish and crsi_1h[i] > 90 and volume_ok:
            short_signal = True
        
        # Path 3: CRSI moderately overbought + both HTF bearish
        if crsi_1h[i] > 75 and strong_bearish and volume_ok:
            short_signal = True
        
        if short_signal:
            desired_signal = -BASE_SIZE
        
        # === CONFLICT: Both signals → go with 1d trend ===
        if long_signal and short_signal:
            if trend_1d_bullish:
                desired_signal = BASE_SIZE
            elif trend_1d_bearish:
                desired_signal = -BASE_SIZE
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
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h HMA still bullish and CRSI not extreme
                if trend_4h_bullish and crsi_1h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h HMA still bearish and CRSI not extreme
                if trend_4h_bearish and crsi_1h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or CRSI extremely overbought
            if trend_4h_bearish or crsi_1h[i] > 85:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or CRSI extremely oversold
            if trend_4h_bullish or crsi_1h[i] < 15:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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