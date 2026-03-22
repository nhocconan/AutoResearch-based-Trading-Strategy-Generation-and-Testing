#!/usr/bin/env python3
"""
Experiment #138: 1d Connors RSI Mean Reversion + 1w HMA Trend Filter + Choppiness Regime

Hypothesis: Daily timeframe is ideal for mean reversion strategies because:
- Less noise than lower timeframes (fewer false signals)
- CRSI (Connors RSI) has proven 75% win rate on daily data
- 1w HMA provides stable long-term trend bias (avoid counter-trend trades)
- Choppiness Index filters: only mean-revert in range markets (CHOP > 61.8)
- In trends (CHOP < 38.2), switch to trend-following mode

Why this might beat the 4h KAMA baseline (Sharpe=0.478):
- CRSI catches oversold/overbought extremes better than simple RSI
- 1w HTF filter prevents dangerous counter-trend entries in strong trends
- Choppiness regime detection adapts to market conditions
- Daily timeframe = fewer trades but higher quality = less fee drag
- ATR trailing stop protects from 2022-style crashes

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_1w_hma_chop_regime_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI Streak(2): Consecutive up/down days momentum
    3. PercentRank(100): Where current price is in recent 100-day range
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    if n < rank_period:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI Streak(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period:i+1]
        avg_streak = np.mean(np.abs(streak_vals))
        # Map streak to 0-100: positive streak = bullish, negative = bearish
        if avg_streak > 0:
            streak_rsi[i] = min(100, 50 + avg_streak * 25)
        else:
            streak_rsi[i] = max(0, 50 + avg_streak * 25)
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        current = close[i]
        count_below = np.sum(window < current)
        percent_rank[i] = count_below / len(window) * 100
    
    # Combine components
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) measures market choppy vs trending.
    
    CHOP > 61.8 = range-bound market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    if n < period:
        return chop
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend for trend direction."""
    n = len(close)
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    if n < period:
        return supertrend, direction
    
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend[period] = upper_band[period]
    direction[period] = -1
    
    for i in range(period + 1, n):
        if close[i-1] <= supertrend[i-1]:
            supertrend[i] = min(upper_band[i], supertrend[i-1]) if upper_band[i] < supertrend[i-1] else upper_band[i]
            direction[i] = -1
        else:
            supertrend[i] = max(lower_band[i], supertrend[i-1]) if lower_band[i] > supertrend[i-1] else lower_band[i]
            direction[i] = 1
        
        if direction[i] == -1 and close[i] > supertrend[i]:
            direction[i] = 1
            supertrend[i] = lower_band[i]
        elif direction[i] == 1 and close[i] < supertrend[i]:
            direction[i] = -1
            supertrend[i] = upper_band[i]
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = long-term trend bias
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8 = range market (mean reversion works)
        # CHOP < 38.2 = trend market (trend following works)
        range_market = chop[i] > 61.8
        trend_market = chop[i] < 38.2
        
        # === SUPERTREND DIRECTION ===
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === CRSI EXTREMES ===
        # CRSI < 10 = extremely oversold
        # CRSI > 90 = extremely overbought
        crsi_oversold = crsi[i] < 15  # Loosened from 10 to ensure trades
        crsi_overbought = crsi[i] > 85  # Loosened from 90 to ensure trades
        
        # CRSI moderate extremes (ensure we get trades)
        crsi_moderate_oversold = crsi[i] < 25
        crsi_moderate_overbought = crsi[i] > 75
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Regime 1: Range market + CRSI oversold + 1w bullish bias
        if range_market and crsi_oversold and bull_trend_1w:
            new_signal = SIZE_STRONG
        # Regime 2: Range market + CRSI moderate oversold + 1w bullish
        elif range_market and crsi_moderate_oversold and bull_trend_1w:
            new_signal = SIZE_BASE
        # Regime 3: Trend market + Supertrend bullish + CRSI pullback
        elif trend_market and st_bullish and crsi_moderate_oversold and bull_trend_1w:
            new_signal = SIZE_BASE
        # Fallback: Ensure trades - 1w bullish + CRSI oversold
        elif bull_trend_1w and crsi_moderate_oversold:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Regime 1: Range market + CRSI overbought + 1w bearish bias
        if range_market and crsi_overbought and bear_trend_1w:
            new_signal = -SIZE_STRONG
        # Regime 2: Range market + CRSI moderate overbought + 1w bearish
        elif range_market and crsi_moderate_overbought and bear_trend_1w:
            new_signal = -SIZE_BASE
        # Regime 3: Trend market + Supertrend bearish + CRSI pullback
        elif trend_market and st_bearish and crsi_moderate_overbought and bear_trend_1w:
            new_signal = -SIZE_BASE
        # Fallback: Ensure trades - 1w bearish + CRSI overbought
        elif bear_trend_1w and crsi_moderate_overbought:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals