#!/usr/bin/env python3
"""
Experiment #242: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + ADX + Choppiness Regime

Hypothesis: After 195+ failed experiments, KAMA (Kaufman Adaptive Moving Average) adapts
better to crypto volatility than HMA. Combined with ADX trend strength filter and
Choppiness Index regime detection, we switch between mean-reversion (chop) and
trend-following (trending) modes. Connors RSI provides responsive entry timing.

Key design:
1. 12h KAMA(10) adaptive trend - responds to volatility changes
2. ADX(14) > 20 filter - only trade when trend has momentum
3. Choppiness Index(14) regime: >61.8 = mean revert, <38.2 = trend follow
4. Connors RSI(3,2,100) for entry timing - catches reversals better than RSI(14)
5. 1d KAMA(21) for macro bias alignment
6. ATR(14) 2.5x trailing stoploss
7. Discrete position sizing: 0.0, ±0.25, ±0.30

TARGET: 25-50 trades/year on 12h, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_chop_crsi_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = close_s.diff(period)
    volatility = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change.abs() / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    tr1 = high_s - low_s
    tr2 = (high_s - close.shift(1)).abs()
    tr3 = (low_s - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    sum_tr = tr.rolling(window=period, min_periods=period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_tr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    return chop.fillna(50).values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (CRSI)."""
    close_s = pd.Series(close)
    
    # RSI component (short period)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)
    
    # Streak RSI component
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
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi = streak_rsi.fillna(50.0)
    
    # Percent Rank component
    def percent_rank(series, window):
        result = pd.Series(np.nan, index=series.index)
        for i in range(window, len(series)):
            rank = (series.iloc[i] > series.iloc[i-window:i]).sum() / window * 100
            result.iloc[i] = rank
        return result
    
    pct_rank = percent_rank(close_s, rank_period)
    pct_rank = pct_rank.fillna(50.0)
    
    # Combine components
    crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    kama_12h = calculate_kama(close, period=10)
    adx_14 = calculate_adx(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate 1d KAMA for macro trend (aligned properly)
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(kama_12h[i]) or np.isnan(adx_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF MACRO BIAS (1d KAMA) ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        macro_bullish = price_above_kama_1d
        macro_bearish = price_below_kama_1d
        
        # === 12h TREND DETECTION (KAMA slope) ===
        kama_slope_bullish = kama_12h[i] > kama_12h[i-5] if i >= 5 else False
        kama_slope_bearish = kama_12h[i] < kama_12h[i-5] if i >= 5 else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20.0
        adx_weak = adx_14[i] <= 20.0
        
        # === CHOPPINESS REGIME ===
        chop_range = chop_14[i] > 61.8  # Mean reversion regime
        chop_trend = chop_14[i] < 38.2  # Trending regime
        chop_neutral = not chop_range and not chop_trend
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_long_entry = crsi[i] < 30.0
        crsi_short_entry = crsi[i] > 70.0
        
        # === DETERMINE DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY/RANGE (mean reversion)
        if chop_range:
            # Long: CRSI oversold + price below KAMA + macro not strongly bearish
            if crsi_oversold and close[i] < kama_12h[i]:
                if macro_bearish:
                    desired_signal = -POSITION_SIZE_HALF  # Counter-trend small
                else:
                    desired_signal = POSITION_SIZE_HALF
            
            # Short: CRSI overbought + price above KAMA + macro not strongly bullish
            elif crsi_overbought and close[i] > kama_12h[i]:
                if macro_bullish:
                    desired_signal = POSITION_SIZE_HALF  # Counter-trend small
                else:
                    desired_signal = -POSITION_SIZE_HALF
        
        # REGIME 2: TRENDING (trend follow)
        elif chop_trend and adx_strong:
            # Long: KAMA slope up + CRSI pullback + macro bullish
            if kama_slope_bullish and crsi_long_entry:
                if macro_bullish:
                    desired_signal = POSITION_SIZE_FULL
                else:
                    desired_signal = POSITION_SIZE_HALF
            
            # Short: KAMA slope down + CRSI pullback + macro bearish
            elif kama_slope_bearish and crsi_short_entry:
                if macro_bearish:
                    desired_signal = -POSITION_SIZE_FULL
                else:
                    desired_signal = -POSITION_SIZE_HALF
        
        # REGIME 3: NEUTRAL (reduced position)
        elif chop_neutral:
            # Only take signals with macro alignment
            if kama_slope_bullish and crsi_long_entry and macro_bullish:
                desired_signal = POSITION_SIZE_HALF
            elif kama_slope_bearish and crsi_short_entry and macro_bearish:
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
        if in_position and position_side > 0 and kama_slope_bearish and adx_strong:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and kama_slope_bullish and adx_strong:
            desired_signal = 0.0
        
        # === MACRO REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bearish and crsi[i] > 60.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bullish and crsi[i] < 40.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if trend still valid ===
        if in_position and desired_signal == 0.0:
            if position_side > 0 and kama_slope_bullish and crsi[i] < 80.0:
                desired_signal = POSITION_SIZE_HALF
            elif position_side < 0 and kama_slope_bearish and crsi[i] > 20.0:
                desired_signal = -POSITION_SIZE_HALF
        
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