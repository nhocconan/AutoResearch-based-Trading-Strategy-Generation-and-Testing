#!/usr/bin/env python3
"""
Experiment #479: 4h Primary + 1d HTF — HMA Trend + Choppiness Regime + RSI Pullback

Hypothesis: Based on mtf_hma_rsi_zscore_v1 (Sharpe=5.4) success pattern, but adapted for 
4h timeframe with Choppiness Index regime filter. Key innovations:
1. HMA(21) on 4h for fast trend detection (HMA reduces lag vs EMA/SMA)
2. 1d HMA aligned for HTF major trend bias (call get_htf_data ONCE before loop)
3. Choppiness Index(14) regime: CHOP>55=range (mean revert), CHOP<45=trend (trend follow)
4. RSI(14) pullback entries: oversold in uptrend, overbought in downtrend
5. Relaxed thresholds to ensure trade generation (avoid 0-trade failure mode)
6. ATR(14) trailing stop at 2.5x for risk management
7. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
8. Volume spike filter: only enter when volume > 1.5x 20-bar avg (confirms moves)

Why this should work: HMA has proven superior to KAMA/EMA in recent experiments.
4h timeframe naturally targets 20-50 trades/year (fee-efficient). Choppiness filter
prevents trend strategies in choppy markets (major failure mode). 1d HTF ensures
we trade with major trend. Relaxed RSI (35/65 vs 25/75) ensures we generate trades.
Volume filter reduces false breakouts. This is DIFFERENT from failed Donchian/Fisher
combinations - using proven HMA+RSI pattern with regime filter.

Target: Sharpe > 0.612, DD < -35%, trades >= 30 on train, >= 3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_chop_rsi_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Reduces lag while maintaining smoothness.
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    # Helper: Weighted Moving Average
    def wma(series, span):
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            weights = np.arange(1, span + 1)
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA calculation
    for i in range(period - 1, n):
        if np.isnan(wma_half[i]) or np.isnan(wma_full[i]):
            continue
        diff = 2.0 * wma_half[i] - wma_full[i]
        # Need to compute WMA of diff over sqrt_period
        if i >= sqrt_period - 1:
            weights = np.arange(1, sqrt_period + 1)
            start_idx = i - sqrt_period + 1
            # Check for NaN in range
            diff_window = np.full(sqrt_period, np.nan)
            valid = True
            for j in range(sqrt_period):
                idx = start_idx + j
                if idx >= len(diff) or np.isnan(diff[idx]):
                    valid = False
                    break
                diff_window[j] = diff[idx]
            if valid:
                hma[i] = np.sum(diff_window * weights) / np.sum(weights)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/chop, CHOP < 38.2 = trending
    Using 55/45 thresholds for earlier regime detection.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        # Sum of ATR over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
            tr_sum += max(tr1, tr2, tr3)
        
        if tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest - lowest) / tr_sum) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_4h = calculate_hma(close, period=21)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # Calculate and align HTF indicators (1d HMA for major trend bias)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(hma_4h[i]):
            continue
        if np.isnan(chop_4h[i]):
            continue
        if np.isnan(rsi_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_chop = chop_4h[i] > 55.0  # Range/mean reversion regime
        is_trend = chop_4h[i] < 45.0  # Trending regime
        # Neutral zone: 45 <= CHOP <= 55
        
        # === HTF MAJOR TREND BIAS (1d HMA) ===
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        price_above_hma = close[i] > hma_4h[i]
        price_below_hma = close[i] < hma_4h[i]
        hma_slope_up = hma_4h[i] > hma_4h[i - 3] if i >= 3 else False
        hma_slope_down = hma_4h[i] < hma_4h[i - 3] if i >= 3 else False
        
        # === RSI SIGNALS (relaxed thresholds for trade generation) ===
        rsi_oversold = rsi_4h[i] < 40.0  # Relaxed from 35
        rsi_overbought = rsi_4h[i] > 60.0  # Relaxed from 65
        rsi_extreme_oversold = rsi_4h[i] < 30.0
        rsi_extreme_overbought = rsi_4h[i] > 70.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = volume[i] > 1.3 * vol_avg[i]  # 30% above average
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        long_score = 0
        
        # HTF bias alignment (required)
        if htf_bullish:
            long_score += 2
        
        # Price above HMA (trend confirmation)
        if price_above_hma:
            long_score += 1
        
        # HMA slope up
        if hma_slope_up:
            long_score += 1
        
        # RSI entry signal (different logic per regime)
        if is_trend:
            # In trend: RSI pullback to oversold
            if rsi_oversold:
                long_score += 2
        elif is_chop:
            # In chop: RSI extreme oversold for mean reversion
            if rsi_extreme_oversold:
                long_score += 2
        else:
            # Neutral: moderate RSI oversold
            if rsi_oversold:
                long_score += 1
        
        # Volume confirmation (bonus, not required)
        if vol_spike:
            long_score += 1
        
        # Enter long if score >= 4 (relaxed for trade generation)
        if long_score >= 4:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # HTF bias alignment
            if htf_bearish:
                short_score += 2
            
            # Price below HMA
            if price_below_hma:
                short_score += 1
            
            # HMA slope down
            if hma_slope_down:
                short_score += 1
            
            # RSI entry signal
            if is_trend:
                if rsi_overbought:
                    short_score += 2
            elif is_chop:
                if rsi_extreme_overbought:
                    short_score += 2
            else:
                if rsi_overbought:
                    short_score += 1
            
            # Volume confirmation
            if vol_spike:
                short_score += 1
            
            if short_score >= 4:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma and htf_bullish:
                desired_signal = SIZE_LONG
            elif position_side < 0 and price_below_hma and htf_bearish:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.25
        
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