#!/usr/bin/env python3
"""
Experiment #1425: 1h Primary + 4h/1d HTF — Relaxed Confluence for Trade Generation

Hypothesis: Previous 1h strategies failed with 0 trades because entry conditions were TOO STRICT.
This version relaxes thresholds while keeping the proven MTF structure:
1. 4h HMA(21) = trend direction (call ONCE before loop)
2. 1d HMA(50) = secondary trend confirmation
3. Choppiness Index(14) = regime detection (>55 range, <45 trend)
4. Connors RSI = entry timing (oversold <30, overbought >70 - RELAXED from <15/>85)
5. Volume filter = only trade when volume > 0.7x 20-bar avg (not 0.8x)
6. ATR(14) trailing stop 2.5x = risk management
7. Position size 0.25 = conservative for 1h volatility

Key changes from failed #1415/#1418/#1420:
- CRSI thresholds: <30/>70 (was <15/>85 - too strict)
- CHOP thresholds: >50/<50 (was >61.8/<38.2 - too strict)
- Volume filter: 0.7x avg (was 0.8x - too strict)
- Removed session filter (was killing trades)
- Added 1d HMA as secondary confirmation (not just 4h)

Target: 40-80 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_4h1d_hma_vol_atr_relaxed_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down bars)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak = 0
        if i > 0:
            if close[i] > close[i-1]:
                j = i
                while j > 0 and close[j] > close[j-1]:
                    streak += 1
                    j -= 1
            elif close[i] < close[i-1]:
                j = i
                while j > 0 and close[j] < close[j-1]:
                    streak -= 1
                    j -= 1
        # Convert streak to RSI-like value (0-100)
        if streak >= 0:
            streak_rsi[i] = min(100.0, streak * 50.0 / streak_period)
        else:
            streak_rsi[i] = max(0.0, 100.0 + streak * 50.0 / streak_period)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0 and not np.any(np.isnan(returns)):
            current_return = returns[-1]
            count_below = np.sum(returns[:-1] < current_return)
            percent_rank[i] = 100.0 * count_below / (len(returns) - 1) if len(returns) > 1 else 50.0
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            
            chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[j] - close[j-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA(21) for primary trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA(50) for secondary trend confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume SMA(20) for volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
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
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (4h HMA) - primary filter ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === SECONDARY TREND (1d HMA) - confirmation ===
        trend_1d_bull = close[i] > hma_1d_aligned[i]
        trend_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 50.0  # Range market (relaxed from 61.8)
        is_trending = chop[i] < 50.0  # Trend market (relaxed from 38.2)
        
        # === CONNORS RSI EXTREMES (mean reversion in chop) ===
        crsi_oversold = crsi[i] < 30.0  # RELAXED from <15
        crsi_overbought = crsi[i] > 70.0  # RELAXED from >85
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.7 * vol_sma[i]  # RELAXED from 0.8x
        
        # === DESIRED SIGNAL - DUAL REGIME LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        # Path 1: Choppy regime + CRSI oversold + 4h bull + volume ok (mean reversion)
        if is_choppy and crsi_oversold and trend_4h_bull and vol_ok:
            desired_signal = BASE_SIZE
        # Path 2: Trending regime + CRSI moderately oversold + 4h bull + 1d bull + volume ok
        elif is_trending and crsi[i] < 40.0 and trend_4h_bull and trend_1d_bull and vol_ok:
            desired_signal = BASE_SIZE
        # Path 3: Very oversold CRSI + 4h bull (strong signal regardless of regime)
        elif crsi[i] < 20.0 and trend_4h_bull and vol_ok:
            desired_signal = BASE_SIZE
        
        # SHORT ENTRIES
        # Path 1: Choppy regime + CRSI overbought + 4h bear + volume ok (mean reversion)
        elif is_choppy and crsi_overbought and trend_4h_bear and vol_ok:
            desired_signal = -BASE_SIZE
        # Path 2: Trending regime + CRSI moderately overbought + 4h bear + 1d bear + volume ok
        elif is_trending and crsi[i] > 60.0 and trend_4h_bear and trend_1d_bear and vol_ok:
            desired_signal = -BASE_SIZE
        # Path 3: Very overbought CRSI + 4h bear (strong signal regardless of regime)
        elif crsi[i] > 80.0 and trend_4h_bear and vol_ok:
            desired_signal = -BASE_SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if abs(desired_signal) >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals