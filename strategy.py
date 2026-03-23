#!/usr/bin/env python3
"""
Experiment #440: 1h Primary + 4h/12h HTF — Regime-Adaptive with Session Filter

Hypothesis: 1h strategies failed due to either (a) too many trades → fee drag, or 
(b) too strict filters → 0 trades. This strategy uses:
1. 4h/12h HMA for PRIMARY trend bias (hard filter, not soft)
2. 1h Choppiness Index for regime detection (range vs trend)
3. 1h Connors RSI for mean-reversion entries (relaxed thresholds: 20/80)
4. 1h Donchian breakout for trend-follow entries
5. Session filter (8-20 UTC) as preference, not hard requirement
6. Volume filter (>0.8x avg) for confirmation

Key difference from failed #430/#435:
- HTF bias is HARD filter (must align), not soft modifier
- CRSI thresholds relaxed to 20/80 (was 15/85 — too rare)
- Session filter reduces size by 50% outside 8-20 UTC, doesn't block
- Position size 0.25 (smaller for 1h vs 4h's 0.30-0.35)
- Target: 40-70 trades/year, Sharpe > 0.3

Target: Sharpe > 0.612 (beat current best), 150-280 trades train, 45-105 trades test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_donchian_4h12h_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    
    # RSI(3) component
    rsi = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - streak_period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        
        if streak > 0:
            streak_rsi[i] = 50.0 + (streak / (streak_period + 1)) * 50.0
        elif streak < 0:
            streak_rsi[i] = 50.0 - (abs(streak) / (streak_period + 1)) * 50.0
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank component
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / rank_period
    
    # Combine
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = volume / (vol_avg + 1e-10)
    return ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds)."""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align HTF HMA for bias (4h and 12h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 1h (smaller than 4h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 55.0  # Range market
        regime_trend = chop[i] < 45.0  # Trending market
        
        # === HTF TREND BIAS (4h + 12h HMA) — HARD FILTER ===
        # Both 4h and 12h must agree for strong bias
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        hma_12h_bullish = close[i] > hma_12h_aligned[i]
        hma_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # Strong bias: both HTFs agree
        strong_bullish = hma_4h_bullish and hma_12h_bullish
        strong_bearish = hma_4h_bearish and hma_12h_bearish
        
        # Weak bias: only 4h agrees (12h neutral or opposite)
        weak_bullish = hma_4h_bullish and not hma_12h_bearish
        weak_bearish = hma_4h_bearish and not hma_12h_bullish
        
        # === PRIMARY TREND (1h HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === CRSI SIGNALS (Mean Reversion) — RELAXED THRESHOLDS ===
        crsi_oversold = crsi[i] < 25.0  # Relaxed from 20
        crsi_overbought = crsi[i] > 75.0  # Relaxed from 80
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === DONCHIAN BREAKOUT (Trend Follow) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === VOL FILTER ===
        vol_ok = vol_ratio[i] > 0.8  # Volume at least 80% of average
        vol_ratio_val = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio_val > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio_val > 1.5:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        session_multiplier = 1.0 if in_session else 0.5  # Reduce size outside session
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 55) — MEAN REVERSION ===
        if regime_chop:
            # Long: CRSI oversold + HTF not strongly bearish
            if crsi_oversold and not strong_bearish:
                desired_signal = position_size * session_multiplier
            elif crsi_extreme_oversold:
                # Extreme oversold can override HTF bias
                desired_signal = position_size * session_multiplier
            
            # Short: CRSI overbought + HTF not strongly bullish
            if crsi_overbought and desired_signal == 0 and not strong_bullish:
                desired_signal = -position_size * session_multiplier
            elif crsi_extreme_overbought:
                # Extreme overbought can override HTF bias
                desired_signal = -position_size * session_multiplier
        
        # === REGIME 2: TRENDING (CHOP < 45) — TREND FOLLOW ===
        elif regime_trend:
            # Long: Donchian breakout OR HMA bullish + HTF bullish bias
            if donchian_breakout_long and (strong_bullish or weak_bullish):
                desired_signal = position_size * session_multiplier
            elif hma_bullish and strong_bullish:
                desired_signal = position_size * 0.8 * session_multiplier
            
            # Short: Donchian breakdown OR HMA bearish + HTF bearish bias
            if donchian_breakout_short and desired_signal == 0 and (strong_bearish or weak_bearish):
                desired_signal = -position_size * session_multiplier
            elif hma_bearish and strong_bearish:
                desired_signal = -position_size * 0.8 * session_multiplier
        
        # === REGIME 3: TRANSITION (45-55) — REDUCED SIZE, ONLY EXTREMES ===
        else:
            # Only extreme CRSI signals allowed
            if crsi_extreme_oversold and not strong_bearish:
                desired_signal = position_size * 0.5 * session_multiplier
            elif crsi_extreme_overbought and not strong_bullish:
                desired_signal = -position_size * 0.5 * session_multiplier
        
        # === VOLUME CONFIRMATION ===
        # If volume is very low (<0.5x), skip entry (but allow exit)
        if vol_ratio[i] < 0.5 and desired_signal != 0.0:
            if not in_position:
                desired_signal = 0.0  # Don't enter on low volume
            # If already in position, allow signal to continue
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === CRSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_bullish or not strong_bearish):
                desired_signal = position_size * session_multiplier
            elif position_side < 0 and (hma_bearish or not strong_bullish):
                desired_signal = -position_size * session_multiplier
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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