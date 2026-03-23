#!/usr/bin/env python3
"""
Experiment #234: 4h Primary + 12h/1d HTF — Donchian Breakout + Connors RSI + ADX Regime

Hypothesis: After #231 failed (Sharpe=-0.642) with simple HMA+RSI, return to proven
combinations from research literature. Donchian breakout captures trend initiation,
Connors RSI (CRSI) provides superior entry timing vs regular RSI (75% win rate in
research), and ADX filters regime to avoid choppy whipsaws.

Key differences from #231:
1. Donchian(20) breakout instead of HMA crossover (proven Sharpe +0.782 on SOL)
2. Connors RSI instead of regular RSI (CRSI<20 long, >80 short vs RSI 40-55)
3. 12h HMA(21) for intermediate trend (not 1d/1w which may be too slow)
4. 1d ADX(14) > 25 for trending regime confirmation
5. Same ATR(14) 2.5x trailing stoploss
6. Discrete sizing: 0.0, ±0.25, ±0.30

TARGET: 25-45 trades/year on 4h, Sharpe > 0.50 on ALL symbols (beat 0.486 baseline)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_crsi_adx_12h1d_atr_v1"
timeframe = "4h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Research shows 75% win rate for mean reversion entries.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI component
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on streak values
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Percent Rank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        lookback = close[i-rank_period:i]
        current = close[i]
        rank = np.sum(lookback < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    # Handle NaN at start
    crsi[:rank_period] = 50.0
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    crsi_3_2_100 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Calculate 12h HMA for intermediate trend (aligned properly)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1d ADX for regime confirmation (aligned properly)
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(crsi_3_2_100[i]):
            signals[i] = 0.0
            continue
        if np.isnan(adx_14[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF REGIME FILTERS ===
        # 12h HMA slope for intermediate trend
        hma_12h_slope_bullish = close[i] > hma_12h_aligned[i]
        hma_12h_slope_bearish = close[i] < hma_12h_aligned[i]
        
        # 1d ADX for trending regime (ADX > 25 = trending, < 20 = ranging)
        adx_1d_trending = adx_1d_aligned[i] > 25.0
        adx_1d_ranging = adx_1d_aligned[i] < 20.0
        
        # === 4h TREND DETECTION (Donchian breakout) ===
        # Price breaking above Donchian upper = bullish breakout
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        # Price breaking below Donchian lower = bearish breakout
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # Price position within channel
        channel_range = donchian_upper[i] - donchian_lower[i]
        if channel_range > 1e-10:
            channel_position = (close[i] - donchian_lower[i]) / channel_range
        else:
            channel_position = 0.5
        
        # === CRSI ENTRY SIGNALS ===
        # CRSI < 20 = oversold (long entry opportunity)
        crsi_oversold = crsi_3_2_100[i] < 20.0
        # CRSI > 80 = overbought (short entry opportunity)
        crsi_overbought = crsi_3_2_100[i] > 80.0
        # CRSI extreme for strong signals
        crsi_extreme_long = crsi_3_2_100[i] < 15.0
        crsi_extreme_short = crsi_3_2_100[i] > 85.0
        
        # === 4h ADX confirmation ===
        adx_4h_trending = adx_14[i] > 20.0
        
        # === DETERMINE DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Donchian breakout + CRSI oversold + trend alignment
        if hma_12h_slope_bullish:
            if donchian_breakout_long and crsi_oversold:
                if adx_1d_trending:
                    desired_signal = POSITION_SIZE_FULL
                elif not adx_1d_ranging:
                    desired_signal = POSITION_SIZE_HALF
            elif crsi_extreme_long and channel_position < 0.3:
                # Mean reversion in bullish trend
                if not adx_1d_ranging:
                    desired_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: Donchian breakdown + CRSI overbought + trend alignment
        elif hma_12h_slope_bearish:
            if donchian_breakout_short and crsi_overbought:
                if adx_1d_trending:
                    desired_signal = -POSITION_SIZE_FULL
                elif not adx_1d_ranging:
                    desired_signal = -POSITION_SIZE_HALF
            elif crsi_extreme_short and channel_position > 0.7:
                # Mean reversion in bearish trend
                if not adx_1d_ranging:
                    desired_signal = -POSITION_SIZE_HALF
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_12h_slope_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_12h_slope_bullish:
            desired_signal = 0.0
        
        # === CRSI EXIT (overbought/oversold reversal) ===
        if in_position and position_side > 0 and crsi_overbought:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_oversold:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if trend still valid ===
        if in_position and desired_signal == 0.0:
            if position_side > 0 and hma_12h_slope_bullish and crsi_3_2_100[i] < 75.0:
                desired_signal = POSITION_SIZE_HALF
            elif position_side < 0 and hma_12h_slope_bearish and crsi_3_2_100[i] > 25.0:
                desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
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