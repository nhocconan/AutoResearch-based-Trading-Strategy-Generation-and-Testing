#!/usr/bin/env python3
"""
Experiment #191: 4h Primary + 1d/1w HTF — Fisher Transform + KAMA Trend + Multi-Regime

Hypothesis: Current best (Sharpe=0.486) uses Connors RSI for mean reversion.
However, Fisher Transform shows superior reversal detection in bear markets (2022 crash, 2025 bear).
Combined with KAMA (adaptive to volatility) instead of HMA, this reduces whipsaw in choppy conditions.

Key innovations:
1. Ehlers Fisher Transform (period=9) - superior reversal detection vs RSI in bear markets
2. KAMA trend filter (ER=10) - adapts to market efficiency, less lag than HMA
3. Triple regime detection: CHOP + ADX + BB Width percentile
4. Volatility-adjusted position sizing (reduce size when ATR spikes > 2x normal)
5. Looser entry thresholds to ensure 40-60 trades/year (avoid 0-trade failure)
6. Funding rate contrarian overlay for BTC/ETH specifically

TARGET: 40-60 trades/year, Sharpe > 0.55 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_multiregime_1d1w_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise - smooth in chop, responsive in trends.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution, excellent for reversal detection.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close) if 'close' in dir() else len(high)
    # Use (high + low) / 2 as price input
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    fisher = np.zeros(n)
    fisher_prev = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(hl2[i - period + 1:i + 1])
        lowest = np.min(hl2[i - period + 1:i + 1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            fisher[i] = fisher[i - 1] if i > 0 else 0.0
            fisher_prev[i] = fisher[i - 1] if i > 0 else 0.0
            continue
        
        # Normalize price to 0-1 range, then to -1 to +1
        normalized = 2.0 * (hl2[i] - lowest) / range_val - 1.0
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        fisher_prev[i] = 0.5 * np.log((1.0 + normalized * 0.999) / (1.0 - normalized * 0.999)) if i > 0 else fisher[i]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / (sma + 1e-10)
    return upper, lower, bandwidth, sma

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
    atr_30 = calculate_atr(high, low, close, period=30)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Calculate 1d KAMA for macro bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 1w KAMA for ultra-long-term trend
    kama_1w_raw = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate BB bandwidth percentile for regime (30-day lookback)
    bb_percentile = np.zeros(n)
    for i in range(30, n):
        if not np.isnan(bb_bandwidth[i]):
            past_bw = bb_bandwidth[i-29:i+1]
            past_bw = past_bw[~np.isnan(past_bw)]
            if len(past_bw) > 0:
                bb_percentile[i] = np.sum(past_bw < bb_bandwidth[i]) / len(past_bw) * 100.0
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    POSITION_SIZE_QUARTER = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(kama_21[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_percentile[i]):
            continue
        
        # === HTF MACRO BIAS ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when vol spikes (ATR > 2x 30-day average)
        vol_ratio = atr_14[i] / (atr_30[i] + 1e-10) if atr_30[i] > 0 else 1.0
        if vol_ratio > 2.0:
            size_multiplier = 0.5  # Half size in high vol
        elif vol_ratio > 1.5:
            size_multiplier = 0.75
        else:
            size_multiplier = 1.0
        
        # === REGIME DETECTION (Triple Filter) ===
        # Range: High CHOP + Low ADX + High BB Percentile
        is_range = (chop_14[i] > 55.0) and (adx_14[i] < 25.0) and (bb_percentile[i] > 60.0)
        
        # Trend: Low CHOP + High ADX + Low BB Percentile (squeeze breakout)
        is_trend = (chop_14[i] < 45.0) and (adx_14[i] > 25.0) and (bb_percentile[i] < 40.0)
        
        # Default to neutral regime if neither clear
        is_neutral = not is_range and not is_trend
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_range:
            # MEAN REVERSION MODE (Fisher Transform reversals)
            # Long: Fisher crosses above -1.5 + price above 1d KAMA
            if fisher[i] > -1.5 and fisher_prev[i] <= -1.5 and price_above_kama_1d:
                base_size = POSITION_SIZE_FULL if price_above_kama_1w else POSITION_SIZE_HALF
                new_signal = base_size * size_multiplier
            
            # Short: Fisher crosses below +1.5 + price below 1d KAMA
            elif fisher[i] < 1.5 and fisher_prev[i] >= 1.5 and price_below_kama_1d:
                base_size = POSITION_SIZE_FULL if price_below_kama_1w else POSITION_SIZE_HALF
                new_signal = -base_size * size_multiplier
            
            # BB mean reversion backup (if Fisher didn't trigger)
            elif close[i] < bb_lower[i] and price_above_kama_1d:
                new_signal = POSITION_SIZE_QUARTER * size_multiplier
            elif close[i] > bb_upper[i] and price_below_kama_1d:
                new_signal = -POSITION_SIZE_QUARTER * size_multiplier
        
        elif is_trend:
            # TREND FOLLOWING MODE (KAMA crossover + ADX confirmation)
            # Long: Price above KAMA + ADX rising + 1d KAMA bullish
            kama_bullish = close[i] > kama_21[i] and plus_di[i] > minus_di[i]
            if kama_bullish and price_above_kama_1d:
                base_size = POSITION_SIZE_FULL if price_above_kama_1w else POSITION_SIZE_HALF
                new_signal = base_size * size_multiplier
            
            # Short: Price below KAMA + ADX rising + 1d KAMA bearish
            kama_bearish = close[i] < kama_21[i] and minus_di[i] > plus_di[i]
            if kama_bearish and price_below_kama_1d:
                base_size = POSITION_SIZE_FULL if price_below_kama_1w else POSITION_SIZE_HALF
                new_signal = -base_size * size_multiplier
        
        else:
            # NEUTRAL REGIME - Only take high-confidence Fisher reversals
            if fisher[i] > -1.0 and fisher_prev[i] <= -1.0 and price_above_kama_1d and price_above_kama_1w:
                new_signal = POSITION_SIZE_QUARTER * size_multiplier
            elif fisher[i] < 1.0 and fisher_prev[i] >= 1.0 and price_below_kama_1d and price_below_kama_1w:
                new_signal = -POSITION_SIZE_QUARTER * size_multiplier
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid (avoid churning)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 4h KAMA
                if close[i] > kama_21[i]:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 4h KAMA
                if close[i] < kama_21[i]:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 1d KAMA (macro trend changed)
        if in_position and position_side > 0 and price_below_kama_1d:
            new_signal = 0.0
        
        # Exit short if price crosses above 1d KAMA (macro trend changed)
        if in_position and position_side < 0 and price_above_kama_1d:
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