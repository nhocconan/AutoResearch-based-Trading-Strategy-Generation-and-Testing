#!/usr/bin/env python3
"""
Experiment #271: 4h Primary + 1d/1w HTF — Regime-Adaptive Strategy

Hypothesis: Previous 4h strategies failed from either over-filtering (#269: 0 trades)
or wrong regime detection (#265: Sharpe=-2.156). This version uses:
- Choppiness Index (14) for regime detection: CHOP>61.8=range, CHOP<38.2=trend
- Connors RSI for mean reversion in choppy regimes (CRSI<15 long, >85 short)
- Donchian(20) breakout for trending regimes (breakout + 1d HMA confirmation)
- 1d HMA(21) for macro bias (soft filter)
- 1w HMA(13) for ultra-long-term trend direction
- ATR(14) 2.5x trailing stoploss
- Position size: 0.30 (discrete levels to minimize fee churn)

KEY INSIGHTS from research:
- CHOP index is the BEST regime filter for bear/range markets (ETH Sharpe +0.923)
- CRSI has 75% win rate on mean reversion entries
- Donchian breakout works in trending regimes but fails in chop (hence regime filter)
- 1d/1w HTF provides macro bias without over-filtering

TARGET: 20-50 trades/year on 4h, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_donchian_regime_1d1w_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    tr_s = pd.Series(tr)
    atr_sum = tr_s.rolling(window=period, min_periods=period).sum().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long opportunity)
    CRSI > 90 = overbought (short opportunity)
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
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_abs = np.abs(streak)
    streak_rsi_gain = np.where(streak > 0, streak_abs, 0)
    streak_rsi_loss = np.where(streak < 0, streak_abs, 0)
    
    streak_avg_gain = pd.Series(streak_rsi_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_rsi_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = np.nan_to_num(rsi_streak, nan=50.0)
    
    # Percent Rank (100)
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100.0 if x.max() > x.min() else 50.0,
        raw=False
    ).values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        crsi = (rsi_short.values + rsi_streak + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-long-term trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, 13)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30  # Discrete level for fee efficiency
    
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
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 61.8  # Range market
        is_trending = chop_14[i] < 38.2  # Trending market
        
        # === MACRO BIAS (1d/1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # --- MEAN REVERSION (Choppy Regime) ---
        if is_choppy:
            # Long: CRSI < 15 + price above 1w HMA (bullish bias)
            if crsi[i] < 15.0 and price_above_hma_1w:
                desired_signal = POSITION_SIZE
            # Short: CRSI > 85 + price below 1w HMA (bearish bias)
            elif crsi[i] > 85.0 and price_below_hma_1w:
                desired_signal = -POSITION_SIZE
        
        # --- TREND FOLLOWING (Trending Regime) ---
        elif is_trending:
            # Long: Donchian breakout + 1d HMA confirmation
            if close[i] > donchian_upper[i] and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            # Short: Donchian breakdown + 1d HMA confirmation
            elif close[i] < donchian_lower[i] and price_below_hma_1d:
                desired_signal = -POSITION_SIZE
        
        # --- TRANSITION ZONE (38.2 <= CHOP <= 61.8) ---
        else:
            # Use softer CRSI thresholds in transition
            if crsi[i] < 20.0 and price_above_hma_1d:
                desired_signal = POSITION_SIZE * 0.5  # Half position
            elif crsi[i] > 80.0 and price_below_hma_1d:
                desired_signal = -POSITION_SIZE * 0.5  # Half position
        
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
        
        # === REGIME CHANGE EXIT ===
        # Exit long if regime changes from choppy to trending bearish
        if in_position and position_side > 0:
            if is_trending and price_below_hma_1d:
                desired_signal = 0.0
        
        # Exit short if regime changes from choppy to trending bullish
        if in_position and position_side < 0:
            if is_trending and price_above_hma_1d:
                desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit in mean reversion) ===
        if in_position and position_side > 0 and crsi[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if still in valid regime
                if is_choppy or (is_trending and price_above_hma_1d):
                    desired_signal = POSITION_SIZE
            elif position_side < 0:
                # Hold short if still in valid regime
                if is_choppy or (is_trending and price_below_hma_1d):
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