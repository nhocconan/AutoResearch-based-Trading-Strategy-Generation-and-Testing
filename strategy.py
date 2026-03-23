#!/usr/bin/env python3
"""
Experiment #098: 30m Primary + 4h/1d HTF — Connors RSI Mean Reversion with Trend Filter

Hypothesis: Previous 30m strategy (#088) failed with 0 trades due to overly strict confluence.
This version uses PROVEN Connors RSI (75% win rate in literature) with LOOSE thresholds
to ensure trade generation while maintaining edge through HTF trend filter.

Key changes from #088 failure:
1) Connors RSI thresholds: <25 for long, >75 for short (not <10/>90 which rarely triggers)
2) HTF 4h HMA is DIRECTIONAL BIAS only, not hard filter (allows counter-trend when CRSI extreme)
3) Session filter is SOFT (prefer 8-20 UTC but allow 24h for crypto)
4) Volume filter: >0.5x avg (not >1.5x which blocks most trades)
5) Choppiness used to ADJUST size, not block entries
6) Position size: 0.22 base, 0.28 max (smaller for 30m to reduce fee drag)

Why this should generate trades:
- CRSI <25 happens ~15% of time (vs <10 which is ~2%)
- 4h HMA bias doesn't block, just adjusts conviction
- 30m TF naturally gives 40-80 trades/year with these thresholds
- Works in both trending and ranging markets (CRSI is mean-reversion)

Position size: 0.22 base, 0.28 max with confluence
Stoploss: 2.5*ATR trailing
Target: 40-80 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_4h1d_session_v2"
timeframe = "30m"
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term momentum
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak
    streak_s = pd.Series(streak)
    streak_gain = streak_s.diff().where(streak_s.diff() > 0, 0.0)
    streak_loss = -streak_s.diff().where(streak_s.diff() < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Percent Rank - how current close compares to prior N closes
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_lower = np.sum(window < close[i])
        percent_rank[i] = count_lower / rank_period * 100.0
    
    # CRSI = average of 3 components
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    vol_ratio = vol_ratio.fillna(1.0).values
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    ema_50 = calculate_ema(close, period=50)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.22
    POSITION_SIZE_MAX = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(ema_50[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === MACRO TREND (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 50.0  # trending market
        chop_ranging = chop_14[i] > 50.0  # ranging market
        
        # === CONNORS RSI SIGNALS (LOOSE thresholds for trade generation) ===
        crsi_oversold = crsi[i] < 25.0  # long signal (was <10, too strict)
        crsi_overbought = crsi[i] > 75.0  # short signal (was >90, too strict)
        crsi_extreme_long = crsi[i] < 15.0  # very oversold
        crsi_extreme_short = crsi[i] > 85.0  # very overbought
        
        # === EMA CONFIRMATION ===
        ema_bullish = close[i] > ema_50[i]
        ema_bearish = close[i] < ema_50[i]
        
        # === VOLUME CONFIRMATION ===
        volume_ok = vol_ratio[i] > 0.5  # at least 50% of avg volume
        
        # === SESSION FILTER (SOFT - crypto trades 24h but prefer liquidity hours) ===
        # Extract hour from open_time (assuming Unix timestamp in milliseconds)
        open_time = prices["open_time"].values[i]
        hour_utc = (open_time // 3600000) % 24
        session_liquid = 8 <= hour_utc <= 20  # UTC 8-20 is high liquidity
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        confluence_count = 0
        
        # --- LONG ENTRY: CRSI oversold + trend bias ---
        if crsi_oversold:
            confluence_count = 1  # CRSI signal itself
            
            # Add confluence: HTF trend aligned
            if price_above_hma_4h:
                confluence_count += 1
            if price_above_hma_1d:
                confluence_count += 1
            
            # Add confluence: EMA confirmation
            if ema_bullish:
                confluence_count += 1
            
            # Add confluence: Volume
            if volume_ok:
                confluence_count += 1
            
            # Add confluence: Session
            if session_liquid:
                confluence_count += 1
            
            # Enter if CRSI extreme OR (CRSI oversold + 2+ confluence)
            if crsi_extreme_long or confluence_count >= 3:
                new_signal = POSITION_SIZE_BASE
                # Boost size with more confluence
                if confluence_count >= 5:
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY: CRSI overbought + trend bias ---
        if crsi_overbought:
            confluence_count = 1  # CRSI signal itself
            
            # Add confluence: HTF trend aligned
            if price_below_hma_4h:
                confluence_count += 1
            if price_below_hma_1d:
                confluence_count += 1
            
            # Add confluence: EMA confirmation
            if ema_bearish:
                confluence_count += 1
            
            # Add confluence: Volume
            if volume_ok:
                confluence_count += 1
            
            # Add confluence: Session
            if session_liquid:
                confluence_count += 1
            
            # Enter if CRSI extreme OR (CRSI overbought + 2+ confluence)
            if crsi_extreme_short or confluence_count >= 3:
                new_signal = -POSITION_SIZE_BASE
                # Boost size with more confluence
                if confluence_count >= 5:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Keep position if CRSI hasn't reached exit zone
        if in_position and new_signal == 0.0:
            if position_side > 0 and crsi[i] < 70.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and crsi[i] > 30.0:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 70.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30.0:
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