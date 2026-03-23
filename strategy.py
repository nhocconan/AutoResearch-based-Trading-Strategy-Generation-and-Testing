#!/usr/bin/env python3
"""
Experiment #295: 1h Primary + 4h/1d HTF — Regime-Adaptive Connors RSI

Hypothesis: Previous 1h strategies (#270, #285, #290) failed due to:
1. RSI 40-60 zone triggers TOO FREQUENTLY (~25% of bars = 2200+ trades/year)
2. No regime filter = trading mean-reversion signals in trending markets
3. No session filter = trading during low-liquidity hours (whipsaw)

This version uses STRICTER confluence (target 30-60 trades/year):
- 4h HMA(21/55) for PRIMARY trend direction
- 1d HMA(21) for MACRO bias (hard filter)
- Connors RSI < 25 or > 75 for entry (MUCH stricter than RSI 40-60)
- Choppiness Index regime: CHOP < 45 = trend (follow), CHOP > 55 = range (revert)
- Volume > 1.0x 20-bar average (stricter than 0.8x)
- Session filter: only 8-20 UTC (high liquidity, less whipsaw)
- ATR(14) 2.0x trailing stoploss (tighter)
- Position size: 0.20 (conservative for 1h fee drag)

KEY DIFFERENCE from #270:
- CRSI < 25/>75 triggers ~5% of bars vs RSI 40-60 at ~25%
- Regime filter blocks 40% of signals (only trade in correct regime)
- Session filter blocks 50% of signals (only 12/24 hours)
- Combined: ~5% * 60% * 50% = 1.5% of bars = ~130 trades/year MAX

TARGET: 30-60 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h1d_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period):
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
    return rsi.fillna(50.0).values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    # Convert streak to RSI-like value (positive streak = bullish)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi = streak_rsi.fillna(50.0).values
    
    # PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_lower = np.sum(window < close[i])
        percent_rank[i] = count_lower / rank_period * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/choppy market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    choppiness = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0  # neutral
    
    return choppiness

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    ratio = vol_s / (vol_ma + 1e-10)
    return ratio.fillna(1.0).values

def get_hour_from_open_time(open_time_arr):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds, convert to hours
    hours = ((open_time_arr / 1000) % 86400) / 3600
    return hours.astype(int)

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
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    hours = get_hour_from_open_time(open_time)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_21_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_55_raw = calculate_hma(df_4h['close'].values, 55)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21_raw)
    hma_4h_55_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_55_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.20  # Conservative for 1h (fee drag control)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_55_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) - HARD FILTER ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA crossover) - PRIMARY FILTER ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_55_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_55_aligned[i]
        
        # === REGIME FILTER (Choppiness Index) ===
        # CHOP < 45 = trending (follow trend)
        # CHOP > 55 = ranging (mean revert)
        # CHOP 45-55 = neutral (no trade)
        regime_trending = chop[i] < 45.0
        regime_ranging = chop[i] > 55.0
        
        # === CONNORS RSI EXTREMES (MUCH STRICTER than RSI 40-60) ===
        # CRSI < 25 = oversold (long opportunity in uptrend)
        # CRSI > 75 = overbought (short opportunity in downtrend)
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.0  # Above average volume
        
        # === SESSION FILTER (8-20 UTC only) ===
        session_active = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bullish + 1d bullish + CRSI oversold + trending regime + volume + session
        if (hma_4h_bullish and price_above_hma_1d and crsi_oversold and 
            regime_trending and volume_confirmed and session_active):
            desired_signal = POSITION_SIZE
        
        # SHORT ENTRY: 4h bearish + 1d bearish + CRSI overbought + trending regime + volume + session
        elif (hma_4h_bearish and price_below_hma_1d and crsi_overbought and 
              regime_trending and volume_confirmed and session_active):
            desired_signal = -POSITION_SIZE
        
        # === MEAN REVERSION IN RANGING MARKET (opposite signals) ===
        # In ranging market, fade extremes against 4h trend
        if regime_ranging and volume_confirmed and session_active:
            if crsi_oversold and hma_4h_bearish:  # Fade short in range
                desired_signal = POSITION_SIZE * 0.5  # Half size for mean reversion
            elif crsi_overbought and hma_4h_bullish:  # Fade long in range
                desired_signal = -POSITION_SIZE * 0.5
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_4h_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_4h_bullish:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and hma_4h_bullish:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and hma_4h_bearish:
                desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals