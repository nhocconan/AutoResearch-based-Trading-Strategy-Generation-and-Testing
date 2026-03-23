#!/usr/bin/env python3
"""
Experiment #173: 1d Primary + 1w HTF — Connors RSI + HMA Trend + Choppiness Regime

Hypothesis: Daily timeframe with weekly HTF bias captures major crypto moves while
filtering noise. Connors RSI (proven 75% win rate) provides precise mean-reversion
entries in the direction of the weekly trend. Choppiness Index adapts position sizing
by regime. This simpler approach should generate consistent trades across ALL symbols
(BTC, ETH, SOL) with positive Sharpe.

KEY IMPROVEMENTS over #172:
1. Connors RSI instead of Fisher — more reliable for crypto mean reversion
2. Simpler entry logic (CRSI extremes + 1w HMA bias) = more trades generated
3. Looser CRSI thresholds (15/85 instead of 10/90) to ensure trade frequency
4. 1d primary timeframe = fewer trades, less fee drag (target 25-40 trades/year)
5. ATR trailing stop at 3.0x for wider stops on daily bars
6. Position size: 0.30 full, 0.20 partial (discrete levels)

TARGET: 25-40 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_hma_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): Fast RSI on price
    RSI(streak, 2): RSI on streak length (consecutive up/down days)
    PercentRank(100): Percentile rank of today's change vs last 100 days
    
    Long when CRSI < 15 (oversold)
    Short when CRSI > 85 (overbought)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on close
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close = rsi_close.fillna(50.0).values
    
    # Streak RSI: count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # PercentRank(100): percentile of today's return vs last 100 days
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current = returns[i]
        if np.isnan(current):
            percent_rank[i] = 50.0
        else:
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                percent_rank[i] = 100.0 * np.sum(valid_window < current) / len(valid_window)
            else:
                percent_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = rsi.fillna(50.0).values
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === REGIME DETECTION ===
        chop_value = chop_14[i]
        is_trending = chop_value < 45.0
        is_ranging = chop_value > 55.0
        
        # === HTF MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D TREND ===
        price_above_hma_21 = close[i] > hma_21[i]
        price_below_hma_21 = close[i] < hma_21[i]
        hma_21_above_50 = hma_21[i] > hma_50[i] if not np.isnan(hma_50[i]) else False
        hma_21_below_50 = hma_21[i] < hma_50[i] if not np.isnan(hma_50[i]) else False
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # CRSI turning up from oversold
        crsi_turning_up = crsi[i] < 30.0 and (i > 0 and crsi[i] > crsi[i-1])
        crsi_turning_down = crsi[i] > 70.0 and (i > 0 and crsi[i] < crsi[i-1])
        
        # === RSI EXTREMES (confirmation) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.5 * vol_avg[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG entries: CRSI oversold + weekly bullish bias
        if crsi_oversold or crsi_turning_up:
            if price_above_hma_1w:
                # Strong long: weekly trend + CRSI oversold
                if is_trending:
                    new_signal = POSITION_SIZE_FULL
                else:
                    new_signal = POSITION_SIZE_HALF
            elif price_above_hma_21 and hma_21_above_50:
                # Moderate long: daily trend + CRSI oversold
                if volume_ok:
                    new_signal = POSITION_SIZE_HALF
        
        # SHORT entries: CRSI overbought + weekly bearish bias
        if crsi_overbought or crsi_turning_down:
            if price_below_hma_1w:
                # Strong short: weekly trend + CRSI overbought
                if is_trending:
                    new_signal = -POSITION_SIZE_FULL
                else:
                    new_signal = -POSITION_SIZE_HALF
            elif price_below_hma_21 and hma_21_below_50:
                # Moderate short: daily trend + CRSI overbought
                if volume_ok:
                    new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 1d HMA
                if price_above_hma_21:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 1d HMA
                if price_below_hma_21:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 1d HMA
        if in_position and position_side > 0 and price_below_hma_21:
            new_signal = 0.0
        
        # Exit short if price crosses above 1d HMA
        if in_position and position_side < 0 and price_above_hma_21:
            new_signal = 0.0
        
        # Exit if weekly bias flips strongly against position
        if in_position and position_side > 0 and price_below_hma_1w:
            new_signal = 0.0
        if in_position and position_side < 0 and price_above_hma_1w:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals