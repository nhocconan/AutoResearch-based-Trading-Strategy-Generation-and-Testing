#!/usr/bin/env python3
"""
Experiment #235: 1h Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + Session Filter

Hypothesis: After 198 failed experiments, lower TF (1h) strategies fail due to:
1. Too many trades → fee drag kills profit
2. No regime filter → mean-revert logic applied in trends (and vice versa)
3. No session filter → low-liquidity hours create false signals

This strategy combines:
1. 4h HMA(16/48) for TREND DIRECTION (not entry trigger)
2. 1d HMA(21) for MACRO BIAS alignment
3. Choppiness Index(14) for REGIME detection (>55=range, <45=trend)
4. Connors RSI for ENTRY TIMING (RSI3 + StreakRSI2 + PercentRank100) / 3
5. Session filter (8-20 UTC) to avoid Asia low-liquidity whipsaw
6. Volume confirmation (>0.8x 20-bar avg)
7. ATR(14) 2.5x trailing stoploss

TARGET: 30-60 trades/year on 1h, Sharpe > 0.45 on ALL symbols
Position sizing: 0.25 full, 0.15 half (conservative for lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_regime_crsi_session_4h1d_atr_v1"
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
    return rsi.fillna(50.0).values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range market, CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        price_range = highest_high - lowest_low
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0, posinf=50.0, neginf=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(returns, 100)) / 3
    
    Long entry: CRSI < 10-20 (oversold)
    Short entry: CRSI > 80-90 (overbought)
    """
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close = rsi_close.fillna(50.0)
    
    # Streak RSI(2)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.fillna(50.0)
    
    # Percent Rank(100)
    returns = close_s.pct_change() * 100
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) * 100 if len(x) >= rank_period else 50.0,
        raw=False
    )
    percent_rank = percent_rank.fillna(50.0)
    
    crsi = (rsi_close.values + rsi_streak.values + percent_rank.values) / 3.0
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h HMA for trend direction (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro bias (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.25
    POSITION_SIZE_HALF = 0.15
    
    # Position tracking (separate from signal output)
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_ma20[i]) or vol_ma20[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high-liquidity hours to avoid Asia session whipsaw
        open_time = prices.iloc[i]['open_time']
        hour_utc = pd.to_datetime(open_time, unit='ms').hour
        in_session = 8 <= hour_utc <= 20
        
        # === 4h TREND DETECTION (HMA crossover for direction) ===
        hma_4h_bullish = hma_16[i] > hma_48[i]
        hma_4h_bearish = hma_16[i] < hma_48[i]
        
        # === 1d MACRO BIAS ===
        price_above_hma_1d = price > hma_1d_aligned[i]
        price_below_hma_1d = price < hma_1d_aligned[i]
        
        # === 4h HTF TREND BIAS ===
        price_above_hma_4h = price > hma_4h_aligned[i]
        price_below_hma_4h = price < hma_4h_aligned[i]
        
        macro_bullish = price_above_hma_1d and price_above_hma_4h
        macro_bearish = price_below_hma_1d and price_below_hma_4h
        macro_neutral = not macro_bullish and not macro_bearish
        
        # === CHOPPINESS REGIME DETECTION ===
        chop_range = chop_14[i] > 55.0  # Range market - favor mean reversion
        chop_trend = chop_14[i] < 45.0  # Trending market - favor trend follow
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_ma20[i]
        
        # === DETERMINE DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bullish + CRSI oversold + regime + session + volume
        if hma_4h_bullish and crsi_oversold and volume_confirmed and in_session:
            if chop_range:
                # Range market: mean reversion long
                if macro_bullish:
                    desired_signal = POSITION_SIZE_FULL
                elif macro_neutral:
                    desired_signal = POSITION_SIZE_HALF
            elif chop_trend:
                # Trending market: pullback long in uptrend
                if macro_bullish or macro_neutral:
                    desired_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: 4h bearish + CRSI overbought + regime + session + volume
        elif hma_4h_bearish and crsi_overbought and volume_confirmed and in_session:
            if chop_range:
                # Range market: mean reversion short
                if macro_bearish:
                    desired_signal = -POSITION_SIZE_FULL
                elif macro_neutral:
                    desired_signal = -POSITION_SIZE_HALF
            elif chop_trend:
                # Trending market: pullback short in downtrend
                if macro_bearish or macro_neutral:
                    desired_signal = -POSITION_SIZE_HALF
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, price)
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if price < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, price)
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if price > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_4h_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_4h_bullish:
            desired_signal = 0.0
        
        # === MACRO REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bullish:
            desired_signal = 0.0
        
        # === CRSI EXIT (overbought for long, oversold for short) ===
        if in_position and position_side > 0 and crsi[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if trend still valid ===
        if in_position and desired_signal == 0.0:
            if position_side > 0 and hma_4h_bullish and crsi[i] < 70.0:
                desired_signal = POSITION_SIZE_HALF
            elif position_side < 0 and hma_4h_bearish and crsi[i] > 30.0:
                desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                highest_since_entry = price if position_side > 0 else float('inf')
                lowest_since_entry = price if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                highest_since_entry = price if position_side > 0 else float('inf')
                lowest_since_entry = price if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals