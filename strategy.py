#!/usr/bin/env python3
"""
Experiment #281: 4h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime

Hypothesis: Previous 4h strategies failed from weak entry signals (#279: HMA+RSI pullback Sharpe=-0.612).
This version uses PROVEN patterns from research:
- Connors RSI (CRSI) for entry timing: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
  Long: CRSI < 15 (oversold), Short: CRSI > 85 (overbought)
- Choppiness Index (CHOP) for regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend
- 1d HMA(21) for macro bias (only trade with daily trend)
- 1w HMA(13) for ultra-long bias (soft filter)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.28 (balanced for 4h volatility)

KEY DIFFERENCES from failed #279:
- Connors RSI instead of regular RSI (more sensitive, 75% win rate in research)
- Choppiness Index regime filter (avoid trend-following in chop, mean-revert in ranges)
- Stricter CRSI thresholds (15/85 vs 40/60) = fewer but higher quality trades
- 1w HMA added for ultra-long bias

TARGET: 25-45 trades/year on 4h, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_1d1w_atr_v1"
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

def calculate_crsi(close):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days (streak length)
    PercentRank: Percentile rank of today's return over last 100 days
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) - short term momentum
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2) - streak of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_rsi = np.zeros(n)
    for i in range(2, n):
        streak_window = streak[max(0, i-2):i+1]
        if len(streak_window) >= 2:
            gains = np.sum(streak_window[streak_window > 0])
            losses = np.abs(np.sum(streak_window[streak_window < 0]))
            if losses == 0:
                streak_rsi[i] = 100.0
            else:
                rs = gains / (losses + 1e-10)
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank (100) - where today's return ranks vs last 100
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / (close[i-1] + 1e-10) * 100
    
    percent_rank = np.zeros(n)
    for i in range(100, n):
        window = returns[i-99:i+1]  # 100 periods including current
        current = returns[i]
        rank = np.sum(window < current) / len(window) * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    for i in range(100, n):
        crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = choppy/range market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    choppiness = np.zeros(n)
    
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
        
        if price_range > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

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
    crsi = calculate_crsi(close)
    choppiness = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-long bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 13)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Conservative for 4h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(choppiness[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === ULTRA-LONG BIAS (1w HMA) - SOFT FILTER ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = choppiness[i] > 61.8  # Range market - mean revert
        is_trending = choppiness[i] < 38.2  # Trend market - trend follow
        
        # === CRSI ENTRY SIGNALS ===
        # Connors RSI extremes for mean reversion
        crsi_oversold = crsi[i] < 15.0  # Strong buy signal
        crsi_overbought = crsi[i] > 85.0  # Strong sell signal
        
        # Moderate CRSI for trend continuation
        crsi_moderate_low = (crsi[i] >= 15.0) and (crsi[i] < 35.0)
        crsi_moderate_high = (crsi[i] > 65.0) and (crsi[i] <= 85.0)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY in CHOPPY regime (mean reversion)
        # CRSI < 15 + price above 1d HMA (with daily trend)
        if is_choppy and crsi_oversold and price_above_hma_1d:
            desired_signal = POSITION_SIZE
        
        # LONG ENTRY in TRENDING regime (trend pullback)
        # CRSI 15-35 + price above 1d HMA + 1w bias aligned
        elif is_trending and crsi_moderate_low and price_above_hma_1d and price_above_hma_1w:
            desired_signal = POSITION_SIZE
        
        # SHORT ENTRY in CHOPPY regime (mean reversion)
        # CRSI > 85 + price below 1d HMA
        if is_choppy and crsi_overbought and price_below_hma_1d:
            desired_signal = -POSITION_SIZE
        
        # SHORT ENTRY in TRENDING regime (trend pullback)
        # CRSI 65-85 + price below 1d HMA + 1w bias aligned
        elif is_trending and crsi_moderate_high and price_below_hma_1d and price_below_hma_1w:
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
        
        # === REGIME CHANGE EXIT ===
        # Exit long if regime switches from choppy to trending against position
        if in_position and position_side > 0 and is_trending and price_below_hma_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and is_trending and price_above_hma_1d:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit on mean reversion) ===
        if in_position and position_side > 0 and crsi[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and price_below_hma_1d:
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