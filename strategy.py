#!/usr/bin/env python3
"""
Experiment #322: 12h Primary + 1d/1w HTF — Adaptive KAMA Regime with CRSI

Hypothesis: 12h timeframe reduces noise and fee drag while maintaining trade quality.
Previous 4h strategies (#321) achieved Sharpe=0.156 but we can improve by:
1. Using 12h primary — fewer false signals, lower fee drag (target 20-40 trades/year)
2. KAMA (Kaufman Adaptive) instead of HMA — adapts to volatility, better in chop
3. Dual HTF bias: 1d HMA for medium-term, 1w HMA for super-macro (bull/bear)
4. LOOSE CRSI thresholds (10/90) to ensure sufficient trades on 12h
5. Wider stoploss (3*ATR) appropriate for 12h volatility
6. Regime detection via Choppiness Index + KAMA efficiency ratio

KEY INSIGHT: 12h bars capture major moves without 4h noise. KAMA ER (Efficiency Ratio)
tells us if market is trending (ER>0.6) or ranging (ER<0.3). Use this for regime switch.

TARGET: 25-40 trades/year on 12h, Sharpe > 0.7 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_regime_crsi_1d1w_bias_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility via Efficiency Ratio (ER).
    ER = |Net Change| / Sum of Absolute Changes (10 periods)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = prior_KAMA + SC * (price - prior_KAMA)
    """
    close_s = pd.Series(close)
    
    # Net change over ER period
    net_change = np.abs(close_s.diff(er_period))
    
    # Sum of absolute changes over ER period
    abs_changes = np.abs(close_s.diff())
    sum_abs_changes = abs_changes.rolling(window=er_period, min_periods=er_period).sum()
    
    # Efficiency Ratio (0 to 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        er = net_change / (sum_abs_changes + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]) or i < er_period:
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama, er.values

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
    # Adjust for direction
    streak_rsi = np.where(delta > 0, streak_rsi, 100 - streak_rsi)
    
    # Percent Rank (100)
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    percent_rank = percent_rank.fillna(50.0)
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    return crsi.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    kama_12h, er_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Calculate and align HTF HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_STRONG = 0.30
    POSITION_SIZE_WEAK = 0.15
    
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
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(er_12h[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d + 1w HMA) — DIRECTIONAL BIAS ONLY ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # Super-macro bias: 1w HMA determines primary direction
        bull_macro = price_above_hma_1w
        bear_macro = price_below_hma_1w
        
        # Medium-term bias: 1d HMA
        bull_medium = price_above_hma_1d
        bear_medium = price_below_hma_1d
        
        # === REGIME DETECTION ===
        # KAMA Efficiency Ratio + Choppiness Index
        er = er_12h[i]
        is_trending_kama = er > 0.5  # KAMA says trending
        is_ranging_kama = er < 0.3   # KAMA says ranging
        
        is_choppy = chop[i] > 55.0   # Choppiness says range
        is_trending_chop = chop[i] < 45.0  # Choppiness says trend
        
        # Combined regime: need agreement
        is_trending = is_trending_kama and is_trending_chop
        is_ranging = is_ranging_kama and is_choppy
        # Default to trend if unclear
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_ranging:
            # RANGE REGIME: Connors RSI Mean Reversion
            # LOOSE thresholds for 12h: CRSI<10 long, >90 short
            if crsi[i] < 10.0:
                if bull_macro:
                    desired_signal = POSITION_SIZE_STRONG  # Bull + oversold = strong long
                elif bear_macro and bull_medium:
                    desired_signal = POSITION_SIZE_WEAK  # Bear macro but 1d bull = weak long
            elif crsi[i] > 90.0:
                if bear_macro:
                    desired_signal = -POSITION_SIZE_STRONG  # Bear + overbought = strong short
                elif bull_macro and bear_medium:
                    desired_signal = -POSITION_SIZE_WEAK  # Bull macro but 1d bear = weak short
        
        else:  # is_trending or neutral
            # TREND REGIME: KAMA trend following + RSI pullback entry
            # Long: KAMA sloping up + RSI pullback to 40-50 + bull macro
            # Short: KAMA sloping down + RSI rally to 50-60 + bear macro
            
            kama_slope = kama_12h[i] - kama_12h[i-3] if i >= 3 else 0
            
            if kama_slope > 0 and bull_macro:
                # KAMA trending up + bull macro
                if 35.0 < rsi_14[i] < 55.0:
                    # RSI pullback entry
                    desired_signal = POSITION_SIZE_STRONG
                elif rsi_14[i] < 35.0:
                    # Deep pullback
                    desired_signal = POSITION_SIZE_STRONG
            
            elif kama_slope < 0 and bear_macro:
                # KAMA trending down + bear macro
                if 45.0 < rsi_14[i] < 65.0:
                    # RSI rally entry
                    desired_signal = -POSITION_SIZE_STRONG
                elif rsi_14[i] > 65.0:
                    # Strong rally
                    desired_signal = -POSITION_SIZE_STRONG
        
        # === STOPLOSS CHECK (3 * ATR trailing for 12h) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit in range regime) ===
        if is_ranging and in_position and position_side > 0 and crsi[i] > 75.0:
            desired_signal = 0.0
        
        if is_ranging and in_position and position_side < 0 and crsi[i] < 25.0:
            desired_signal = 0.0
        
        # === KAMA REVERSAL EXIT ===
        if in_position and position_side > 0:
            if kama_slope < 0 and bear_medium:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if kama_slope > 0 and bull_medium:
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Hold position — maintain current signal
            if position_side > 0:
                desired_signal = POSITION_SIZE_STRONG
            elif position_side < 0:
                desired_signal = -POSITION_SIZE_STRONG
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
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