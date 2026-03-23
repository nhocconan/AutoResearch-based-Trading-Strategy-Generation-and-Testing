#!/usr/bin/env python3
"""
Experiment #203: 1d Primary + 1w HTF — Dual Regime (Mean Reversion + Trend Follow)

Hypothesis: Daily timeframe with weekly HTF bias provides cleanest signals with minimal noise.
Key insight from failed experiments: 1d strategies need LOOSER entry thresholds to generate
adequate trades (20-50/year). Previous 1d attempts (#197, #202) failed due to:
1. Entry conditions too strict (Donchian + CRSI + Choppiness all required)
2. Negative Sharpe from whipsaw in 2022 bear market

This experiment uses:
1. Choppiness Index regime switch (CHOP > 55 = range, CHOP < 45 = trend)
2. Connors RSI for mean reversion entries (thresholds 20/80, not 10/90)
3. 1w KAMA for macro directional bias (aligned via mtf_data helper)
4. Daily KAMA for trend following entries
5. Asymmetric sizing: 0.30 with trend, 0.20 counter-trend
6. ATR trailing stoploss (2.5x) for risk management

Why this should work on 1d:
- Fewer false signals than lower TF
- Weekly bias filters out counter-trend trades in strong moves
- Dual regime adapts to market conditions (range vs trend)
- Looser CRSI thresholds ensure 30-50 trades/year

TARGET: Sharpe > 0.5 on ALL symbols, 30-50 trades/year, DD < -30%
Position sizing: 0.0, ±0.20, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_kama_chop_1w_v1"
timeframe = "1d"
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

def calculate_kama(close, er_period=10, fast_sc=2/11, slow_sc=2/101):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    kama = np.zeros(n)
    
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i-er_period])
        noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or tr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

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
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) - fast momentum
    rsi3 = calculate_rsi(close, period=3)
    
    # RSI of streak - consecutive up/down
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_positive = np.maximum(streak, 0)
    streak_negative = np.abs(np.minimum(streak, 0))
    
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        avg_gain = np.mean(streak_positive[i-streak_period+1:i+1])
        avg_loss = np.mean(streak_negative[i-streak_period+1:i+1])
        if avg_loss < 1e-10:
            streak_rsi[i] = 100.0
        else:
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / (avg_loss + 1e-10)))
    
    # PercentRank - where current close ranks vs last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(max(3, streak_period, rank_period), n):
        crsi[i] = (rsi3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    kama_14 = calculate_kama(close, er_period=10)
    hma_21 = calculate_hma(close, period=21)
    
    # Calculate 1w KAMA for macro trend (aligned properly)
    kama_1w_raw = calculate_kama(df_1w['close'].values, er_period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate 1w HMA for additional trend confirmation
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(kama_14[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === HTF MACRO BIAS (1w KAMA + HMA) ===
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bullish: both KAMA and HMA agree
        macro_bullish = price_above_kama_1w and price_above_hma_1w
        macro_bearish = price_below_kama_1w and price_below_hma_1w
        macro_neutral = not macro_bullish and not macro_bearish
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = chop_14[i] > 55.0  # Ranging market
        is_trend = chop_14[i] < 45.0  # Trending market
        # Neutral zone 45-55: use trend following bias
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_range:
            # MEAN REVERSION MODE (CRSI extremes) - LOOSER thresholds for more trades
            # Long: CRSI < 25 (oversold) + macro not strongly bearish
            if crsi[i] < 25:
                if macro_bearish:
                    new_signal = 0.0  # Skip counter-macro trades in range
                elif macro_bullish:
                    new_signal = POSITION_SIZE_FULL  # With macro trend
                else:
                    new_signal = POSITION_SIZE_HALF  # Neutral macro
            
            # Short: CRSI > 75 (overbought) + macro not strongly bullish
            elif crsi[i] > 75:
                if macro_bullish:
                    new_signal = 0.0  # Skip counter-macro trades in range
                elif macro_bearish:
                    new_signal = -POSITION_SIZE_FULL  # With macro trend
                else:
                    new_signal = -POSITION_SIZE_HALF  # Neutral macro
        
        elif is_trend:
            # TREND FOLLOWING MODE (KAMA/HMA + CRSI filter)
            # Long: Price above KAMA(14) + CRSI not overbought (< 70) + macro bullish/neutral
            if close[i] > kama_14[i] and crsi[i] < 70:
                if macro_bullish:
                    new_signal = POSITION_SIZE_FULL
                elif macro_neutral:
                    new_signal = POSITION_SIZE_HALF
                # Skip if macro bearish (counter-trend)
            
            # Short: Price below KAMA(14) + CRSI not oversold (> 30) + macro bearish/neutral
            elif close[i] < kama_14[i] and crsi[i] > 30:
                if macro_bearish:
                    new_signal = -POSITION_SIZE_FULL
                elif macro_neutral:
                    new_signal = -POSITION_SIZE_HALF
                # Skip if macro bullish (counter-trend)
        
        else:
            # NEUTRAL ZONE (45-55 CHOP): Use simpler trend following
            # Long: Price above HMA(21) + CRSI < 60
            if close[i] > hma_21[i] and crsi[i] < 60:
                if not macro_bearish:
                    new_signal = POSITION_SIZE_HALF
            
            # Short: Price below HMA(21) + CRSI > 40
            elif close[i] < hma_21[i] and crsi[i] > 40:
                if not macro_bullish:
                    new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and conditions still valid (reduces churn)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not extremely overbought and price above KAMA
                if crsi[i] < 80 and close[i] > kama_14[i] * 0.98:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not extremely oversold and price below KAMA
                if crsi[i] > 20 and close[i] < kama_14[i] * 1.02:
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
        
        # === MACRO TREND REVERSAL EXIT ===
        # Exit long if 1w macro turns strongly bearish
        if in_position and position_side > 0 and macro_bearish:
            new_signal = 0.0
        
        # Exit short if 1w macro turns strongly bullish
        if in_position and position_side < 0 and macro_bullish:
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
                # Position flip
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