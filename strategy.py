#!/usr/bin/env python3
"""
Experiment #699: 4h Primary + 1d HTF — Dual Regime (CHOP + CRSI + Donchian)

Hypothesis: After 600+ failed strategies, the clearest pattern is:
1. Single-regime strategies fail because BTC/ETH switch between trend and range
2. 4h timeframe needs HTF (1d) filter to avoid whipsaw (seen in #689, #691, #694 failures)
3. Connors RSI has literature backing for 75% win rate on mean-reversion
4. Choppiness Index is the best regime meta-filter (CHOP>61.8=range, <38.2=trend)

This strategy uses DUAL REGIME approach:
- RANGE regime (1d CHOP > 55): Mean-revert using Connors RSI extremes
- TREND regime (1d CHOP < 45): Breakout using Donchian + 1d HMA filter
- Transition zone (45-55): Hold previous regime (hysteresis to prevent flipping)

Key innovations vs failed attempts:
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven in literature
- CHOP(14) on 1d for regime — avoids 4h noise
- 1d HMA(21) for trend bias — smoother than EMA
- Donchian(10) for breakouts — faster than 20-period
- Asymmetric sizing: 0.30 for trend, 0.25 for mean-revert (trend has higher conviction)

Position sizing: 0.25-0.30 discrete (Rule 4 compliant)
Target: 25-45 trades/year on 4h (within Rule 10 limits)
Stoploss: 2.5*ATR trailing

Why this might beat Sharpe=0.520:
- Dual regime adapts to market conditions (single-regime strategies fail in 2022-2024)
- CRSI extremes catch reversals better than simple RSI (literature: 75% win rate)
- 1d CHOP filter prevents trend-following in chop (major loss source)
- Hysteresis prevents rapid regime flipping (fee reduction)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dualregime_chop_crsi_donchian_1d_v1"
timeframe = "4h"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Market is chopping/ranging
    - CHOP < 38.2: Market is trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness calculation
    range_hl = hh - ll
    chop = 100.0 * np.log10(atr_sum / (range_hl + 1e-10)) / np.log10(period)
    
    return chop.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Components:
    1. RSI(3) of close — short-term momentum
    2. RSI(2) of streak — streak duration (consecutive up/down days)
    3. PercentRank(100) — where current price ranks vs last 100 bars
    
    Signals:
    - Long: CRSI < 15 (oversold extreme)
    - Short: CRSI > 85 (overbought extreme)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) of close
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI(2) of streak
    # Streak: consecutive up (+1) or down (-1) days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100)
    # Where does current close rank vs last 100 bars?
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100.0
    )
    
    # Combine into CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def calculate_donchian(high, low, period=10):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2.0
    
    return upper.values, lower.values, middle.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators
    chop_1d = calculate_choppiness_index(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF indicators to 4h (auto shift(1) for completed bars)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_TREND = 0.30      # Higher conviction for trend breakouts
    SIZE_MEANREV = 0.25    # Slightly lower for mean-reversion
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # Regime hysteresis tracking (prevents rapid flipping)
    prev_regime = 0  # 0=neutral, 1=trend, 2=range
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(chop_1d_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D CHOP REGIME (with hysteresis) ===
        chop_val = chop_1d_aligned[i]
        
        # Hysteresis: trend needs CHOP<45 to enter, >55 to exit range
        if chop_val < 45.0:
            regime = 1  # Trending
        elif chop_val > 55.0:
            regime = 2  # Range
        else:
            regime = prev_regime  # Keep previous regime in transition zone
        
        prev_regime = regime
        is_trend = (regime == 1)
        is_range = (regime == 2)
        
        # === 1D HMA TREND BIAS ===
        hma_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-3] if i >= 3 else False
        hma_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-3] if i >= 3 else False
        price_above_hma = close[i] > hma_1d_aligned[i]
        price_below_hma = close[i] < hma_1d_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 25.0   # Long entry threshold (looser for more trades)
        crsi_overbought = crsi[i] > 75.0  # Short entry threshold
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above prev high
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below prev low
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGE REGIME: Mean-Reversion with CRSI ---
        if is_range:
            # Long: CRSI oversold + price near/near 1d HMA support
            if crsi_oversold:
                if price_above_hma or (close[i] > hma_1d_aligned[i] * 0.98):
                    new_signal = SIZE_MEANREV
            
            # Short: CRSI overbought + price near/near 1d HMA resistance
            if crsi_overbought:
                if price_below_hma or (close[i] < hma_1d_aligned[i] * 1.02):
                    new_signal = -SIZE_MEANREV
        
        # --- TREND REGIME: Breakout with Donchian + HMA filter ---
        elif is_trend:
            # Long: Donchian breakout + 1d HMA bullish
            if donchian_breakout_long and hma_slope_bull and price_above_hma:
                new_signal = SIZE_TREND
            
            # Short: Donchian breakout + 1d HMA bearish
            if donchian_breakout_short and hma_slope_bear and price_below_hma:
                new_signal = -SIZE_TREND
        
        # === HOLD POSITION LOGIC ===
        # If we're in a position and no new signal, keep the position
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Long position: trail stop below highest price since entry
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Short position: trail stop above lowest price since entry
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
        # Close long if 1d HMA turns bearish and price breaks below
        if in_position and position_side > 0:
            if hma_slope_bear and price_below_hma and is_trend:
                new_signal = 0.0
        
        # Close short if 1d HMA turns bullish and price breaks above
        if in_position and position_side < 0:
            if hma_slope_bull and price_above_hma and is_trend:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip (reverse)
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            # Exit position
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals