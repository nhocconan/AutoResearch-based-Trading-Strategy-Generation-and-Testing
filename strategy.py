#!/usr/bin/env python3
"""
Experiment #233: 1d Primary + 1w HTF — Donchian Breakout + Connors RSI + Choppiness Regime

Hypothesis: After 195+ failed experiments with complex Fisher/KAMA regimes, return to 
proven higher-timeframe patterns. 1d timeframe with 1w macro bias should reduce noise 
and whipsaw while maintaining 25-40 trades/year.

Key components:
1. Donchian(20) breakout for trend direction (proven on daily)
2. Connors RSI (RSI3 + RSI_Streak2 + PercentRank100) / 3 for entry timing
3. Choppiness Index(14) regime filter: >61.8 = range (mean revert), <38.2 = trend
4. 1w HMA(21) for macro bias alignment
5. ATR(14) 2.5x trailing stoploss
6. Discrete position sizing: 0.0, ±0.25, ±0.30

TARGET: 25-40 trades/year on 1d, Sharpe > 0.50 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_crsi_chop_1w_atr_v1"
timeframe = "1d"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of current close within lookback period
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, period=rsi_period)
    
    # Streak calculation (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # PercentRank(100): percentile of current close within last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        lookback = close[i-rank_period+1:i+1]
        rank = np.sum(lookback < close[i]) / rank_period
        percent_rank[i] = rank * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    
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
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 1w HMA for macro trend (aligned properly)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking (separate from signal output)
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === HTF MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        macro_bullish = price_above_hma_1w
        macro_bearish = price_below_hma_1w
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop[i] > 61.8  # Range/mean reversion
        trending_regime = chop[i] < 38.2  # Trend following
        neutral_regime = not choppy_regime and not trending_regime
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_breakout_long = close[i] >= donchian_upper[i-1]  # Break above previous high
        donchian_breakout_short = close[i] <= donchian_lower[i-1]  # Break below previous low
        
        # === CONNORS RSI ENTRY SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Extreme oversold for long
        crsi_overbought = crsi[i] > 85.0  # Extreme overbought for short
        crsi_neutral_long = 20.0 <= crsi[i] <= 50.0  # Pullback zone for long
        crsi_neutral_short = 50.0 <= crsi[i] <= 80.0  # Pullback zone for short
        
        # === DETERMINE DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY LOGIC
        if trending_regime or neutral_regime:
            # Trend following: Donchian breakout + macro bias + CRSI confirmation
            if donchian_breakout_long and macro_bullish and crsi[i] < 70.0:
                desired_signal = POSITION_SIZE_FULL
            elif donchian_breakout_long and crsi_neutral_long:
                desired_signal = POSITION_SIZE_HALF
        elif choppy_regime:
            # Mean reversion: CRSI oversold + price near Donchian lower
            if crsi_oversold and close[i] <= donchian_lower[i] * 1.02:
                desired_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY LOGIC
        if trending_regime or neutral_regime:
            # Trend following: Donchian breakout + macro bias + CRSI confirmation
            if donchian_breakout_short and macro_bearish and crsi[i] > 30.0:
                desired_signal = -POSITION_SIZE_FULL
            elif donchian_breakout_short and crsi_neutral_short:
                desired_signal = -POSITION_SIZE_HALF
        elif choppy_regime:
            # Mean reversion: CRSI overbought + price near Donchian upper
            if crsi_overbought and close[i] >= donchian_upper[i] * 0.98:
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
        if in_position and position_side > 0:
            # Exit if Donchian breaks down or macro turns bearish
            if close[i] < donchian_lower[i-1] or macro_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit if Donchian breaks up or macro turns bullish
            if close[i] > donchian_upper[i-1] or macro_bullish:
                desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if trend still valid ===
        if in_position and desired_signal == 0.0:
            if position_side > 0 and close[i] > donchian_upper[i-1] * 0.95 and crsi[i] < 80.0:
                desired_signal = POSITION_SIZE_HALF
            elif position_side < 0 and close[i] < donchian_lower[i-1] * 1.05 and crsi[i] > 20.0:
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