#!/usr/bin/env python3
"""
Experiment #1428: 30m Primary + 4h/1d HTF — Relaxed Regime + CRSI + HMA Trend

Hypothesis: 30m timeframe with RELAXED entry conditions will generate sufficient trades
while maintaining quality. Previous 30m/1h strategies failed with 0 trades because:
1. Too many AND conditions (CHOP + CRSI + HTF + volume + session = nothing triggers)
2. Thresholds too strict (CRSI < 15, CHOP > 61.8 rarely align)
3. HTF too slow (1w HMA doesn't match 30m entry timing)

Solution for 30m:
1. 4h HMA(21) = primary trend filter (faster than 1d/1w, matches 30m entries better)
2. 1d HMA(21) = secondary confirmation (only for strong signals)
3. CRSI(3,2,100) with RELAXED thresholds: < 25 oversold, > 75 overbought (not 15/85)
4. Choppiness Index: > 50 = choppy (mean revert), < 45 = trending (breakout)
5. Volume filter: OPTIONAL boost (not required) - volume > 0.7x 20-bar avg
6. Session filter: 8-20 UTC preferred but NOT required (allows overnight trades)
7. ATR(14) trailing stop 2.5x = risk management

Why this should work:
- 4h HMA provides trend direction without being too slow (unlike 1w)
- Relaxed CRSI thresholds ensure trades actually trigger
- OR logic for entry paths (choppy mean-revert OR trending breakout)
- Position size 0.25 (conservative for 30m frequency)

Target: 40-80 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_relaxed_regime_crsi_4h1d_hma_atr_v1"
timeframe = "30m"
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels for entry trigger"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for primary trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for secondary confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_20_upper[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (4h HMA) - primary filter ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === SECONDARY CONFIRMATION (1d HMA) - optional boost ===
        daily_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 50.0  # Range market (relaxed from 61.8)
        is_trending = chop[i] < 45.0  # Trend market (relaxed from 38.2)
        
        # === CONNORS RSI EXTREMES (relaxed thresholds) ===
        crsi_oversold = crsi[i] < 25.0  # Relaxed from 15
        crsi_overbought = crsi[i] > 75.0  # Relaxed from 85
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_20_long = close[i] > donchian_20_upper[i-1] if i > 0 else False
        breakout_20_short = close[i] < donchian_20_lower[i-1] if i > 0 else False
        
        # === VOLUME FILTER (optional boost, not required) ===
        vol_ok = not np.isnan(vol_avg[i]) and vol_avg[i] > 0 and volume[i] > 0.7 * vol_avg[i]
        
        # === SESSION FILTER (8-20 UTC preferred but not required) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        session_ok = 8 <= hour_utc <= 20
        
        # === DESIRED SIGNAL - RELAXED DUAL REGIME LOGIC ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # LONG ENTRIES
        # Path 1: Choppy regime + CRSI oversold + 4h trend bull (mean reversion)
        if is_choppy and crsi_oversold and trend_bull:
            desired_signal = BASE_SIZE
            signal_strength = 1.0
            if daily_bull:
                signal_strength = 1.2
            if vol_ok:
                signal_strength = 1.3
        
        # Path 2: Trending regime + Donchian breakout + 4h trend bull
        elif is_trending and breakout_20_long and trend_bull:
            desired_signal = BASE_SIZE
            signal_strength = 1.0
            if daily_bull:
                signal_strength = 1.2
            if vol_ok:
                signal_strength = 1.3
        
        # Path 3: Extreme CRSI oversold + 4h trend bull (strong signal, any regime)
        elif crsi_extreme_oversold and trend_bull:
            desired_signal = BASE_SIZE
            signal_strength = 1.4
        
        # Path 4: Very extreme CRSI + session OK (highest conviction)
        elif crsi[i] < 10.0 and trend_bull and session_ok:
            desired_signal = BASE_SIZE
            signal_strength = 1.5
        
        # SHORT ENTRIES
        # Path 1: Choppy regime + CRSI overbought + 4h trend bear
        elif is_choppy and crsi_overbought and trend_bear:
            desired_signal = -BASE_SIZE
            signal_strength = 1.0
            if daily_bear:
                signal_strength = 1.2
            if vol_ok:
                signal_strength = 1.3
        
        # Path 2: Trending regime + Donchian breakout + 4h trend bear
        elif is_trending and breakout_20_short and trend_bear:
            desired_signal = -BASE_SIZE
            signal_strength = 1.0
            if daily_bear:
                signal_strength = 1.2
            if vol_ok:
                signal_strength = 1.3
        
        # Path 3: Extreme CRSI overbought + 4h trend bear
        elif crsi_extreme_overbought and trend_bear:
            desired_signal = -BASE_SIZE
            signal_strength = 1.4
        
        # Path 4: Very extreme CRSI + session OK
        elif crsi[i] > 90.0 and trend_bear and session_ok:
            desired_signal = -BASE_SIZE
            signal_strength = 1.5
        
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
            signal_strength = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
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