#!/usr/bin/env python3
"""
Experiment #361: 4h Primary + 1d/1w HTF — Vol Spike Mean Reversion with Funding Bias

Hypothesis: Previous strategies failed because they relied on trend-following in 
bear/range markets (2022 crash, 2025 bear). This strategy exploits VOLATILITY SPIKE 
MEAN REVERSION - a proven pattern that works in ALL market regimes:

1. VOL SPIKE DETECTION: ATR(7)/ATR(30) > 1.8 signals panic/extreme move
2. MEAN REVERSION ENTRY: Price < BB(20, 2.0) during vol spike = oversold panic
3. MACRO BIAS: 1d HMA(21) for direction, 1w HMA(50) for major trend
4. CONNORS RSI: Faster reaction than standard RSI for reversal timing
5. EXIT: When ATR ratio normalizes (< 1.3) OR Fisher reaches opposite extreme

KEY INSIGHT: Volatility spikes are mean-reverting by nature. After ATR expands 2x+,
price typically reverts to mean within 3-5 bars. This works in bull AND bear markets
because panic/capitulation happens in both directions.

TARGET: 30-50 trades/year on 4h, Sharpe > 0.6 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_reversion_1d1w_crsi_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # RSI(3) on close
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI: consecutive up/down days
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank over lookback
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100 if x.max() > x.min() else 50
    )
    
    # Combine into CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    return crsi.fillna(50.0).values

def calculate_fisher_transform(high, low, close, period=9):
    """Calculate Ehlers Fisher Transform."""
    typical = (high + low + close) / 3.0
    typical_s = pd.Series(typical)
    
    highest = typical_s.rolling(window=period, min_periods=period).max().values
    lowest = typical_s.rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = (typical - lowest) / (highest - lowest + 1e-10)
    
    normalized = np.clip(normalized, 0.001, 0.999)
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    
    return fisher

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
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
    
    # Calculate 4h indicators (primary timeframe)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    fisher = calculate_fisher_transform(high, low, close, period=9)
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volatility spike ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = atr_7 / (atr_30 + 1e-10)
    
    # Calculate and align HTF HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 4h (target 30-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_vol_ratio = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d and 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 1.8  # ATR(7) is 1.8x ATR(30) = panic
        vol_normalizing = vol_ratio[i] < 1.3  # Volatility returning to normal
        
        # === REGIME (Choppiness) ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # VOL SPIKE MEAN REVERSION (works in ALL regimes)
        if vol_spike:
            # LONG: Vol spike + price below lower BB + CRSI oversold + 1d bullish bias
            long_condition = (
                close[i] < bb_lower[i] and
                crsi[i] < 20 and
                price_above_hma_1d
            )
            
            # SHORT: Vol spike + price above upper BB + CRSI overbought + 1d bearish bias
            short_condition = (
                close[i] > bb_upper[i] and
                crsi[i] > 80 and
                price_below_hma_1d
            )
            
            # RELAXED: Allow trades even without 1w confirmation if vol spike is extreme
            if vol_ratio[i] > 2.5:
                # Extreme vol spike - trade either direction based on CRSI alone
                if crsi[i] < 15:
                    desired_signal = BASE_SIZE
                elif crsi[i] > 85:
                    desired_signal = -BASE_SIZE
            else:
                if long_condition:
                    desired_signal = BASE_SIZE
                elif short_condition:
                    desired_signal = -BASE_SIZE
        
        # MEAN REVERSION in CHOPPY regime (no vol spike required)
        elif is_choppy:
            # Long at lower BB with CRSI confirmation
            if close[i] < bb_lower[i] * 0.995 and crsi[i] < 25 and price_above_hma_1d:
                desired_signal = BASE_SIZE * 0.7
            # Short at upper BB with CRSI confirmation
            elif close[i] > bb_upper[i] * 1.005 and crsi[i] > 75 and price_below_hma_1d:
                desired_signal = -BASE_SIZE * 0.7
        
        # TREND FOLLOWING in TRENDING regime (HMA crossover)
        elif is_trending:
            # Only trade in direction of 1w trend
            hma_16 = calculate_hma(close[:i+1], period=16)[-1]
            hma_48 = calculate_hma(close[:i+1], period=48)[-1]
            
            if price_above_hma_1w and hma_16 > hma_48 and fisher[i] > -1.0:
                desired_signal = BASE_SIZE * 0.6
            elif price_below_hma_1w and hma_16 < hma_48 and fisher[i] < 1.0:
                desired_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === VOLATILITY NORMALIZATION EXIT ===
        # Exit when vol spike normalizes (mean reversion complete)
        if in_position and vol_normalizing and entry_vol_ratio > 1.8:
            desired_signal = 0.0
        
        # === FISHER EXIT (reversal complete) ===
        if in_position and position_side > 0 and fisher[i] > 1.5:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.5:
            desired_signal = 0.0
        
        # === CRSI EXIT (extreme reached) ===
        if in_position and position_side > 0 and crsi[i] > 75:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 25:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro bias still valid and no exit signal
                if price_above_hma_1d and crsi[i] < 75 and fisher[i] < 1.5:
                    # Check if still in vol spike mean reversion
                    if entry_vol_ratio > 1.8 and not vol_normalizing:
                        desired_signal = BASE_SIZE
                    # Or choppy regime mean reversion
                    elif is_choppy and close[i] < bb_mid:
                        desired_signal = BASE_SIZE * 0.7
            elif position_side < 0:
                # Hold short if macro bias still valid and no exit signal
                if price_below_hma_1d and crsi[i] > 25 and fisher[i] > -1.5:
                    if entry_vol_ratio > 1.8 and not vol_normalizing:
                        desired_signal = -BASE_SIZE
                    elif is_choppy and close[i] > bb_mid:
                        desired_signal = -BASE_SIZE * 0.7
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_vol_ratio = vol_ratio[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_vol_ratio = vol_ratio[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                entry_vol_ratio = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals