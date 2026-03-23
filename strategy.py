#!/usr/bin/env python3
"""
Experiment #140: 1h Primary + 4h/12h HTF — Choppiness Regime + Connors RSI Pullback

Hypothesis: Previous 1h strategies failed due to (1) zero trades from over-filtering,
or (2) too many trades causing fee drag. This uses PROVEN components:

1) 4h HMA(21) for trend direction — only trade pullbacks in trend direction
2) Choppiness Index(14) for regime — CHOP>55=range(mean revert), CHOP<45=trend(follow)
3) Connors RSI for entry timing — CRSI<20 long, CRSI>80 short (proven 75% win rate)
4) Session filter 8-20 UTC — only trade during high-volume hours
5) Volume confirmation — volume > 0.8x 20-bar avg (not too strict)
6) ATR(14) 2.5x trailing stop — protects capital

Key improvements over failed #135:
- LOOSEN entry conditions (CRSI<25/>75 not <20/>80)
- Add regime-adaptive logic (range vs trend)
- Ensure trades on ALL symbols (BTC/ETH/SOL)
- Target 40-80 trades/year (not <10, not >200)

Position size: 0.25 base, 0.30 max with confluence
Stoploss: 2.5*ATR trailing
Timeframe: 1h with 4h trend bias
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_pullback_4h_session_v1"
timeframe = "1h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) by E.W. Dreiss.
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.maximum(price_range, 1e-10)  # avoid div by zero
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) by Larry Connors.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3) = fast momentum
    RSI(streak) = streak strength (consecutive up/down days)
    PercentRank = where current price ranks vs last 100 bars
    """
    close_s = pd.Series(close)
    
    # RSI(3) - fast momentum
    delta = close_s.diff()
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_fast = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI - count consecutive up/down bars
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak
    streak_s = pd.Series(streak)
    streak_gain = np.maximum(streak_s.diff(), 0)
    streak_loss = -np.minimum(streak_s.diff(), 0)
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank - where does current close rank vs last 100?
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100,
        raw=False
    )
    
    # Combine into CRSI
    crsi = (rsi_fast + rsi_streak + percent_rank) / 3.0
    return crsi.values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    # Calculate 1h HMA for additional trend filter
    hma_1h_21 = calculate_hma(close, period=21)
    hma_1h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        if np.isnan(hma_1h_21[i]) or np.isnan(hma_1h_50[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_ok = volume_ratio > 0.8  # relaxed threshold
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1h TREND FILTER ===
        hma_1h_bullish = hma_1h_21[i] > hma_1h_50[i]
        hma_1h_bearish = hma_1h_21[i] < hma_1h_50[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # range market
        is_trending = chop_value < 45.0  # trend market
        # neutral zone 45-55: use both signals
        
        # === CONNORS RSI SIGNALS ===
        crsi_value = crsi[i]
        crsi_oversold = crsi_value < 25.0  # long entry
        crsi_overbought = crsi_value > 75.0  # short entry
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 4h trend up + 1h trend up + CRSI oversold
        # In range market: mean revert (can enter against short-term trend)
        # In trend market: only enter with trend
        long_condition_trend = price_above_hma_4h and hma_1h_bullish and crsi_oversold
        long_condition_range = price_above_hma_4h and crsi_oversold  # relaxed in range
        
        if in_session and volume_ok:
            if is_trending:
                if long_condition_trend:
                    new_signal = POSITION_SIZE_BASE
                    if volume_ratio > 1.3:
                        new_signal = POSITION_SIZE_MAX
            elif is_ranging:
                if long_condition_range:
                    new_signal = POSITION_SIZE_BASE
            else:  # neutral zone
                if long_condition_trend or long_condition_range:
                    new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY ---
        short_condition_trend = price_below_hma_4h and hma_1h_bearish and crsi_overbought
        short_condition_range = price_below_hma_4h and crsi_overbought
        
        if in_session and volume_ok:
            if is_trending:
                if short_condition_trend:
                    new_signal = -POSITION_SIZE_BASE
                    if volume_ratio > 1.3:
                        new_signal = -POSITION_SIZE_MAX
            elif is_ranging:
                if short_condition_range:
                    new_signal = -POSITION_SIZE_BASE
            else:  # neutral zone
                if short_condition_trend or short_condition_range:
                    new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if still in position and trend intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if 4h trend still up
                if price_above_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if 4h trend still down
                if price_below_hma_4h:
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
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h:
                new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi_value > 70.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi_value < 30.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals