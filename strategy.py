#!/usr/bin/env python3
"""
Experiment #636: 12h Primary + 1d HTF — Dual Regime (Chop/Trend) + Connors RSI + Donchian

Hypothesis: Based on proven 12h patterns (Choppiness+Connors RSI showed ETH Sharpe +0.923,
Donchian+HMA+RSI showed SOL Sharpe +0.782), this strategy uses a dual-regime approach:
- RANGE regime (CHOP > 50): Mean reversion via Connors RSI extremes
- TREND regime (CHOP < 50): Breakout following 1d HMA trend direction

Key insights from 562 failed strategies:
1. Single-regime strategies fail in mixed markets (2022 crash + 2025 bear)
2. Connors RSI works better than standard RSI for mean reversion (75% win rate)
3. Choppiness Index is the best regime filter for bear/range markets
4. 12h + 1d MTF combination reduces whipsaw vs single timeframe
5. Too many filters = 0 trades (see #628, #632, #635 with Sharpe=0.000)
6. Need simpler entry logic to ensure 20-50 trades/year on 12h

Why this might beat Sharpe=0.520:
- Regime-aware: different logic for chop vs trend (adaptive to market state)
- Connors RSI (3 components) catches oversold/overbought better than RSI(14)
- 1d HMA slope filter keeps us on right side of major moves
- Donchian(20) breakout confirms momentum in trend regime
- Conservative sizing (0.28) + 2.5*ATR stop controls drawdown
- Fewer conflicting filters = more trades (target 25-45/year on 12h)

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 12h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_donchian_1d_v1"
timeframe = "12h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_connors_rsi(close, period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term RSI for quick reversals
    RSI_Streak(2): RSI of consecutive up/down days
    PercentRank(100): Where current price ranks vs last 100 bars
    
    CRSI < 10 = oversold (long signal)
    CRSI > 90 = overbought (short signal)
    """
    close_s = pd.Series(close)
    
    # RSI(3) - very short term
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain_3 = gain.ewm(span=3, min_periods=3, adjust=False).mean()
    avg_loss_3 = loss.ewm(span=3, min_periods=3, adjust=False).mean()
    
    rs_3 = avg_gain_3 / (avg_loss_3 + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs_3))
    
    # RSI of Streak (consecutive up/down days)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=2, min_periods=2, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=2, min_periods=2, adjust=False).mean()
    
    rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Percent Rank (where current close ranks vs last 100)
    percent_rank = close_s.rolling(window=period, min_periods=period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100.0
        if x.max() > x.min() else 50.0
    )
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel upper and lower bands.
    Upper = highest high over period
    Lower = lowest low over period
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 100)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    hma_12h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range_regime = chop_14[i] > 50.0  # Choppy/range market
        is_trend_regime = chop_14[i] <= 50.0  # Trending market
        
        # === 1D TREND BIAS (HMA slope over 3 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-3] if i >= 3 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-3] if i >= 3 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 12H HMA SLOPE ===
        hma_12h_slope_bull = hma_12h[i] > hma_12h[i-2] if i >= 2 else False
        hma_12h_slope_bear = hma_12h[i] < hma_12h[i-2] if i >= 2 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        if is_range_regime:
            # MEAN REVERSION: Connors RSI extreme oversold in range market
            if crsi_extreme_oversold or (crsi_oversold and price_above_hma_1d):
                new_signal = POSITION_SIZE
        else:
            # TREND FOLLOWING: Breakout + trend alignment
            if hma_1d_slope_bull and price_above_hma_1d:
                if donchian_breakout_up and hma_12h_slope_bull:
                    new_signal = POSITION_SIZE
                elif crsi_oversold and hma_12h_slope_bull:
                    # Pullback entry in uptrend
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        if is_range_regime:
            # MEAN REVERSION: Connors RSI extreme overbought in range market
            if crsi_extreme_overbought or (crsi_overbought and price_below_hma_1d):
                new_signal = -POSITION_SIZE
        else:
            # TREND FOLLOWING: Breakout + trend alignment
            if hma_1d_slope_bear and price_below_hma_1d:
                if donchian_breakout_down and hma_12h_slope_bear:
                    new_signal = -POSITION_SIZE
                elif crsi_overbought and hma_12h_slope_bear:
                    # Pullback entry in downtrend
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
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals