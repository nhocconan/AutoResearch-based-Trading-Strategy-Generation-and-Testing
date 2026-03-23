#!/usr/bin/env python3
"""
Experiment #645: 1h Primary + 4h/1d HTF — Regime-Adaptive CRSI + HMA Trend + Session Filter

Hypothesis: 1h timeframe needs VERY strict filters to avoid fee drag (target 30-80 trades/year).
This strategy uses 1d HMA for major trend direction, 4h Choppiness Index for regime detection,
and 1h Connors RSI for precise entry timing. Session filter (8-20 UTC) avoids low-liquidity periods.

Key insights from failed 1h/30m strategies (#635, #638, #640):
1. Too many confluence filters = 0 trades (need balance)
2. Session filter alone killed all trades in #638
3. Volume filter must be lenient (>0.5x avg, not >1.5x)
4. RSI extremes (not narrow ranges) generate actual entries

Why this might beat Sharpe=0.520:
- 1d HMA slope keeps us on right side of major moves (simpler than 1w)
- Choppiness regime adapts: mean-revert in range, trend-follow in trends
- Connors RSI (3-period) catches short-term extremes better than RSI(14)
- Session filter only narrows to 8-20 UTC (12 hours, not 4 hours)
- Discrete sizing (0.25) minimizes fee churn
- ATR stoploss protects against reversals

Position sizing: 0.25 discrete (per Rule 4, max 0.40)
Target: 40-70 trades/year on 1h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma_4h1d_session_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI(streak): Consecutive up/down days
    PercentRank: Where current close ranks in last 100 bars
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - very short term
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down moves
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak (convert to positive for RSI calc)
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = np.where(streak >= 0, rsi_streak, 100 - rsi_streak)
    
    # Percent Rank - where current close ranks in last rank_period bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = count_below / rank_period * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Avoid division by zero
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    chop = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
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
    
    # Calculate 1d HMA for major trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h Choppiness Index for regime detection
    chop_4h = calculate_choppiness(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    
    # Calculate 1h indicators
    hma_1h = calculate_hma(close, period=21)
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Extract hour from open_time for session filter
    # open_time is in milliseconds since epoch
    hours = pd.to_datetime(prices['open_time'], unit='ms').hour.values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(chop_4h_aligned[i]) or np.isnan(crsi_1h[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high-liquidity hours
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        # Volume must be at least 0.6x average (lenient to allow trades)
        volume_ok = vol_ratio[i] >= 0.6
        
        # === 1D TREND BIAS (HMA slope over 5 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H CHOPPINESS REGIME ===
        chop_high = chop_4h_aligned[i] > 55.0  # Range/choppy market
        chop_low = chop_4h_aligned[i] < 45.0   # Trending market
        
        # === 1H CRSI EXTREMES ===
        crsi_oversold = crsi_1h[i] < 25.0
        crsi_overbought = crsi_1h[i] > 75.0
        crsi_extreme = crsi_oversold or crsi_overbought
        
        # === 1H HMA SLOPE (3 bars) ===
        hma_1h_slope_bull = hma_1h[i] > hma_1h[i-3] if i >= 3 else False
        hma_1h_slope_bear = hma_1h[i] < hma_1h[i-3] if i >= 3 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 1d bull trend + CRSI oversold + regime-appropriate ---
        # In trending regime (CHOP < 45): follow trend on pullback
        # In ranging regime (CHOP > 55): mean revert from oversold
        if in_session and volume_ok:
            if hma_1d_slope_bull and price_above_hma_1d:
                if crsi_oversold and hma_1h_slope_bull:
                    # Add regime confirmation
                    if chop_low or (chop_high and close[i] < hma_1h[i] * 0.98):
                        new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 1d bear trend + CRSI overbought + regime-appropriate ---
        elif in_session and volume_ok:
            if hma_1d_slope_bear and price_below_hma_1d:
                if crsi_overbought and hma_1h_slope_bear:
                    # Add regime confirmation
                    if chop_low or (chop_high and close[i] > hma_1h[i] * 1.02):
                        new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals