#!/usr/bin/env python3
"""
Experiment #311: 4h Primary + 1d/1w HTF — Fisher Transform + Dual Regime Strategy

Hypothesis: Current best (#301) uses CRSI+Donchian but may miss reversal entries in bear markets.
Fisher Transform (Ehlers) excels at catching reversals in bear market rallies (2022, 2025).
Combined with Choppiness regime filter + CRSI confirmation + 1d/1w macro bias.

KEY IMPROVEMENTS over #301:
- Fisher Transform (period=9): Long when crosses above -1.5, Short when crosses below +1.5
- Dual HTF: 1d HMA for intermediate bias, 1w HMA for macro regime
- LOOSE entry thresholds to ensure ≥30 trades/year on 4h
- ATR 2.5x trailing stoploss on all positions
- Position size: 0.28 (conservative for 4h frequency)

REGIME LOGIC:
- CHOP > 55: Range market → Fisher + CRSI mean reversion entries
- CHOP < 45: Trend market → Donchian breakout + HMA alignment
- CHOP 45-55: Neutral → Use Fisher reversals only (most reliable)

MACRO FILTER (1d + 1w HMA):
- Long only if: price > 1d HMA OR 1w HMA sloping up
- Short only if: price < 1d HMA OR 1w HMA sloping down
- This prevents fighting the macro trend

TARGET: 25-40 trades/year, Sharpe > 0.65 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_regime_crsi_1d1w_atr_v1"
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
        rsi_short = 100.0 - (100.0 / (1.0 + rs))
    rsi_short = rsi_short.fillna(50.0)
    
    # RSI Streak (2)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    gain_streak = streak_s.diff().clip(lower=0)
    loss_streak = (-streak_s.diff()).clip(lower=0)
    avg_gain_streak = gain_streak.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss_streak = loss_streak.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_gain_streak / (avg_loss_streak + 1e-10)
        streak_rsi = 100.0 - (100.0 / (1.0 + rs_streak))
    streak_rsi = streak_rsi.fillna(50.0)
    streak_rsi = np.where(delta > 0, streak_rsi, 100 - streak_rsi)
    
    # Percent Rank (100)
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    percent_rank = percent_rank.fillna(50.0)
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    return crsi.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.67
    """
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Normalize price position within range
    with np.errstate(divide='ignore', invalid='ignore'):
        x = 0.67 * (close - lowest) / (highest - lowest + 1e-10) - 0.67
        x = np.clip(x, -0.99, 0.99)  # Prevent log domain errors
        fisher = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
    
    # Signal line is previous Fisher value
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    
    # Calculate and align 1d HMA for intermediate bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1w HMA slope (for macro trend direction)
    hma_1w_slope = np.zeros(n)
    for i in range(2, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]):
            hma_1w_slope[i] = hma_1w_aligned[i] - hma_1w_aligned[i-1]
        else:
            hma_1w_slope[i] = 0.0
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Conservative for 4h frequency
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA + 1w HMA slope) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_1w_sloping_up = hma_1w_slope[i] > 0
        hma_1w_sloping_down = hma_1w_slope[i] < 0
        
        # Long bias: price > 1d HMA OR 1w HMA sloping up
        long_bias = price_above_hma_1d or hma_1w_sloping_up
        # Short bias: price < 1d HMA OR 1w HMA sloping down
        short_bias = price_below_hma_1d or hma_1w_sloping_down
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        # 45-55 = neutral
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Fisher + CRSI Mean Reversion
            # LONG: Fisher crosses above -1.5 + CRSI < 25 + long bias
            if fisher[i] > -1.5 and fisher_signal[i] <= -1.5 and crsi[i] < 25.0 and long_bias:
                desired_signal = POSITION_SIZE
            # SHORT: Fisher crosses below +1.5 + CRSI > 75 + short bias
            elif fisher[i] < 1.5 and fisher_signal[i] >= 1.5 and crsi[i] > 75.0 and short_bias:
                desired_signal = -POSITION_SIZE
        
        elif is_trending:
            # TREND REGIME: Donchian Breakout with RSI confirmation
            # LONG: Price breaks Donchian upper + RSI > 50 + long bias
            if close[i] > donchian_upper[i-1] and rsi_14[i] > 50.0 and long_bias:
                desired_signal = POSITION_SIZE
            # SHORT: Price breaks Donchian lower + RSI < 50 + short bias
            elif close[i] < donchian_lower[i-1] and rsi_14[i] < 50.0 and short_bias:
                desired_signal = -POSITION_SIZE
        
        else:  # Neutral regime (45-55)
            # Use Fisher reversals only (most reliable in uncertain markets)
            if fisher[i] > -1.5 and fisher_signal[i] <= -1.5 and long_bias:
                desired_signal = POSITION_SIZE
            elif fisher[i] < 1.5 and fisher_signal[i] >= 1.5 and short_bias:
                desired_signal = -POSITION_SIZE
        
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
        
        # === MACRO BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1d and not hma_1w_sloping_up:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d and not hma_1w_sloping_down:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit in range regime) ===
        if is_choppy and in_position and position_side > 0 and crsi[i] > 70.0:
            desired_signal = 0.0
        
        if is_choppy and in_position and position_side < 0 and crsi[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_1d or hma_1w_sloping_up):
                desired_signal = POSITION_SIZE
            elif position_side < 0 and (price_below_hma_1d or hma_1w_sloping_down):
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