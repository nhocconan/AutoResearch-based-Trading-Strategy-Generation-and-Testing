#!/usr/bin/env python3
"""
Experiment #599: 12h Connors RSI + Vol Spike Mean Reversion with 1d/1w HMA Filter

Hypothesis: After 530+ failures, the pattern is clear:
1. Pure trend strategies fail in bear/range markets (2022 crash, 2025 bear)
2. Mean reversion works better BUT needs HTF trend filter to avoid catching falling knives
3. Connors RSI (CRSI) has 75% win rate in literature for short-term reversals
4. Vol spike detection (ATR(7)/ATR(30) > 2.0) captures panic extremes
5. 12h timeframe needs LOOSE entry thresholds to generate sufficient trades

Why this should beat #593 (Sharpe=-0.281):
- #593 used Choppiness regime filter (lagging, few signals)
- This uses Connors RSI + Vol Spike (faster, more responsive)
- Looser RSI thresholds (RSI3<20 vs RSI14<30) = more entry opportunities
- 1d + 1w HMA dual filter = better trend alignment than single HTF
- Target: 40-60 trades/train, 10-15 trades/test (vs #593's ~20 trades)

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing (slightly wider for 12h noise)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_connors_rsi_vol_spike_dual_htf_hma_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Fast momentum
    RSI(Streak): Consecutive up/down days
    PercentRank: Where current close ranks vs last 100 closes
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak values (treat streak as price)
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine components
    crsi = (rsi_fast + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_vol_spike_ratio(atr_short, atr_long):
    """Calculate volatility spike ratio (ATR short / ATR long)."""
    ratio = atr_short / atr_long
    ratio = np.where(np.isnan(ratio) | np.isinf(ratio), 0.0, ratio)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    rsi_3 = calculate_rsi(close, 3)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    vol_spike_ratio = calculate_vol_spike_ratio(atr_7, atr_30)
    
    # Bollinger Bands for additional mean reversion confirmation
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.5 * bb_std
    bb_lower = bb_mid - 2.5 * bb_std
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_ENTRY = 0.28
    SIZE_EXIT = 0.0
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(rsi_3[i]) or np.isnan(vol_spike_ratio[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === HTF TREND BIAS (1d + 1w HMA) ===
        # Bullish: price > 1d HMA AND 1d HMA > 1w HMA
        # Bearish: price < 1d HMA AND 1d HMA < 1w HMA
        bull_bias = (close[i] > hma_1d_aligned[i]) and (hma_1d_aligned[i] > hma_1w_aligned[i])
        bear_bias = (close[i] < hma_1d_aligned[i]) and (hma_1d_aligned[i] < hma_1w_aligned[i])
        neutral_bias = not bull_bias and not bear_bias
        
        # === VOL SPIKE DETECTION ===
        vol_spike = vol_spike_ratio[i] > 1.8  # ATR(7) > 1.8x ATR(30)
        
        # === CONNORS RSI EXTREMES (loose for 12h) ===
        crsi_oversold = crsi[i] < 20  # Very oversold
        crsi_overbought = crsi[i] > 80  # Very overbought
        
        # === RSI(3) EXTREMES (faster signal) ===
        rsi3_oversold = rsi_3[i] < 15
        rsi3_overbought = rsi_3[i] > 85
        
        # === BOLLINGER BAND EXTREMES ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Multiple confirmations for mean reversion
        # Need: CRSI oversold OR (RSI3 oversold + BB lower) + Vol spike helps
        long_confidence = 0
        if crsi_oversold:
            long_confidence += 2
        if rsi3_oversold:
            long_confidence += 1
        if bb_oversold:
            long_confidence += 1
        if vol_spike:
            long_confidence += 1
        
        # Enter long if confidence >= 3 AND not strongly bearish HTF
        if long_confidence >= 3:
            if bull_bias or neutral_bias:
                new_signal = SIZE_ENTRY
            elif bear_bias and vol_spike:
                # Even in bear trend, vol spike + extreme oversold = bounce play
                new_signal = SIZE_ENTRY * 0.5  # Half size in counter-trend
        
        # SHORT ENTRY: Multiple confirmations for mean reversion
        short_confidence = 0
        if crsi_overbought:
            short_confidence += 2
        if rsi3_overbought:
            short_confidence += 1
        if bb_overbought:
            short_confidence += 1
        if vol_spike:
            short_confidence += 1
        
        # Enter short if confidence >= 3 AND not strongly bullish HTF
        if short_confidence >= 3:
            if bear_bias or neutral_bias:
                new_signal = -SIZE_ENTRY
            elif bull_bias and vol_spike:
                # Even in bull trend, vol spike + extreme overbought = pullback play
                new_signal = -SIZE_ENTRY * 0.5  # Half size in counter-trend
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === MEAN REVERSION EXIT (CRSI crosses back to neutral) ===
        mean_reversion_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi[i] > 60:
                # Long position: exit when CRSI recovers to neutral
                mean_reversion_exit = True
            if position_side < 0 and crsi[i] < 40:
                # Short position: exit when CRSI drops to neutral
                mean_reversion_exit = True
        
        # Apply stoploss or mean reversion exit
        if stoploss_triggered or mean_reversion_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals