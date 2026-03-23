#!/usr/bin/env python3
"""
Experiment #401: 4h Primary + 1d HTF — Simplified HMA Trend + Connors RSI Pullback

Hypothesis: The #399 strategy failed (Sharpe=-0.019) due to overly complex regime logic
with too many confluence requirements. This version simplifies to proven patterns:
1. 4h HMA(21/50) for primary trend direction
2. 1d HMA(21) for HTF bias filter (load ONCE before loop)
3. Connors RSI (CRSI) for entry timing - proven 75% win rate in research
4. Asymmetric sizing: 0.30 in strong trend, 0.20 in weak trend
5. ATR(14) trailing stop at 2.5x for longs, 2.0x for shorts

Why this should beat Sharpe=0.612:
- Simpler entry conditions = more trades (target 30-50/year on 4h)
- Connors RSI proven in research notes (ETH Sharpe +0.923 with CRSI)
- Based on mtf_hma_rsi_zscore_v1 baseline (Sharpe=5.4) but adapted for 4h
- Less restrictive than #399's Donchian + CHOP + RSI combo
- Discrete sizing (0.0, ±0.20, ±0.30) minimizes fee churn

Target: Sharpe > 0.612, 30-50 trades/year, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_crsi_pullback_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(period // 2, 1)
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = max(int(np.sqrt(period)), 1)
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank for Connors RSI."""
    n = len(close)
    pr = np.zeros(n)
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        pr[i] = 100.0 * count_below / (period - 1)
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.zeros(n)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(streak_period, n):
        abs_streak = np.abs(streak[i-period+1:i+1]).mean() if i >= streak_period else np.abs(streak[i])
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + abs_streak * 10)
        else:
            streak_rsi[i] = max(0, 50 - abs_streak * 10)
    
    # Percent Rank
    pr = calculate_percent_rank(close, pr_period)
    
    # Combine
    crsi = (rsi_short + streak_rsi + pr) / 3.0
    return crsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma_slope(hma, lookback=5):
    """Calculate HMA slope (positive = uptrend, negative = downtrend)."""
    n = len(hma)
    slope = np.zeros(n)
    for i in range(lookback, n):
        slope[i] = (hma[i] - hma[i-lookback]) / (hma[i-lookback] + 1e-10) * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    hma_slope_21 = calculate_hma_slope(hma_21, lookback=5)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    atr_median = np.nanmedian(atr_14[100:])
    
    signals = np.zeros(n)
    BASE_SIZE_STRONG = 0.30  # 30% in strong trend
    BASE_SIZE_WEAK = 0.20    # 20% in weak trend
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(hma_slope_21[i]):
            continue
        
        # === HTF BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        hma_slope_positive = hma_slope_21[i] > 0.5  # Strong uptrend
        hma_slope_negative = hma_slope_21[i] < -0.5  # Strong downtrend
        
        # === TREND STRENGTH ===
        is_strong_trend = (hma_bullish and hma_slope_positive) or (hma_bearish and hma_slope_negative)
        position_size = BASE_SIZE_STRONG if is_strong_trend else BASE_SIZE_WEAK
        
        # === VOL FILTER (reduce size in extreme vol) ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = position_size * 0.5
        elif vol_ratio > 1.8:
            position_size = position_size * 0.75
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 10 = deeply oversold (long opportunity)
        # CRSI > 90 = deeply overbought (short opportunity)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP
        if price_above_hma_1d and hma_bullish:
            if crsi_oversold:
                # Deep pullback in uptrend - high probability long
                desired_signal = position_size
            elif rsi_14[i] < 35 and hma_slope_positive:
                # RSI pullback with positive slope
                desired_signal = position_size
        
        # SHORT SETUP
        if price_below_hma_1d and hma_bearish:
            if crsi_overbought:
                # Deep rally in downtrend - high probability short
                desired_signal = -position_size
            elif rsi_14[i] > 65 and hma_slope_negative:
                # RSI rally with negative slope
                desired_signal = -position_size
        
        # === STOPLOSS CHECK (Asymmetric: tighter on shorts) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d:
            desired_signal = 0.0
        
        # === CRSI EXIT (extreme reached - take profit) ===
        if in_position and position_side > 0 and crsi[i] > 80:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1d and hma_bullish:
                desired_signal = position_size
            elif position_side < 0 and price_below_hma_1d and hma_bearish:
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals