#!/usr/bin/env python3
"""
Experiment #461: 4h Primary + 1d/1w HTF — KAMA Trend + Fisher Transform + Choppiness Regime

Hypothesis: Based on research showing KAMA (Kaufman Adaptive Moving Average) adapts better
to crypto volatility regimes than simple EMA/HMA. Combined with Ehlers Fisher Transform
for precise reversal entries (better than RSI in bear markets) and Choppiness Index for
regime detection. Key innovations:
1. KAMA(10) adaptive trend — slows in chop, speeds in trends
2. Ehlers Fisher Transform(9) — catches reversals at extremes (-1.5/+1.5 thresholds)
3. Choppiness Index(14) regime switch — CHOP>61.8=range, <38.2=trend
4. 1d HMA(21) for medium-term bias, 1w HMA(21) for ultra-long bias
5. Volume spike confirmation for breakouts (vol>1.5x median)
6. Asymmetric sizing: 0.30 long in bull, 0.20 short in bull (and vice versa)
7. ATR(14) trailing stop at 2.5x for risk management

Target: Sharpe > 0.612, 20-50 trades/year, DD < -35%
Timeframe: 4h (proven best for swing trading with manageable fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_volume_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    er = np.nan_to_num(er, nan=0.0)
    
    # Calculate smoothing constant
    sc = (er * (2.0/(fast_period + 1) - 2.0/(slow_period + 1)) + 2.0/(slow_period + 1)) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """Calculate Ehlers Fisher Transform."""
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            x = (2.0 * (high[i] + low[i]) / 2.0 - highest - lowest) / price_range
            x = np.clip(x, -0.999, 0.999)
            
            fisher_val = 0.5 * np.log((1.0 + x) / (1.0 - x))
            
            # Smooth with previous value
            if i > period and not np.isnan(fisher[i-1]):
                fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
            else:
                fisher[i] = fisher_val
            
            if i > 0 and not np.isnan(fisher[i-1]):
                trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
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
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h indicators (primary timeframe)
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_30 = calculate_kama(close, period=30, fast_period=2, slow_period=30)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Calculate median volume for spike detection
    vol_median = np.nanmedian(volume[100:])
    if np.isnan(vol_median) or vol_median <= 0:
        vol_median = np.nanmean(volume[100:])
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE_LONG = 0.30  # 30% for longs in bull regime
    BASE_SIZE_SHORT = 0.20  # 20% for shorts in bull regime (asymmetric)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(kama_10[i]) or np.isnan(kama_30[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 61.8  # Range market - mean reversion
        regime_trend = chop[i] < 38.2  # Trending market - breakout
        
        # === HTF TREND BIAS (1d and 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (KAMA crossover) ===
        kama_bullish = kama_10[i] > kama_30[i]
        kama_bearish = kama_10[i] < kama_30[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > fisher_trigger[i] if not np.isnan(fisher_trigger[i]) else False
        fisher_cross_down = fisher[i] < fisher_trigger[i] if not np.isnan(fisher_trigger[i]) else False
        
        # === VOLUME SPIKE ===
        vol_spike = volume[i] > 1.5 * vol_median if vol_median > 0 else False
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ASYMMETRIC POSITION SIZING ===
        if price_above_hma_1w:  # Bull regime
            size_long = BASE_SIZE_LONG
            size_short = BASE_SIZE_SHORT
        else:  # Bear regime
            size_long = BASE_SIZE_SHORT
            size_short = BASE_SIZE_LONG
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 61.8) — MEAN REVERSION ===
        if regime_chop:
            # Long: Fisher oversold + RSI oversold + HTF not bearish
            if fisher_oversold and rsi_oversold:
                signal_strength = 2
                if price_above_hma_1d:
                    signal_strength += 1
                if price_above_hma_1w:
                    signal_strength += 1
                
                if signal_strength >= 2:
                    desired_signal = size_long * (0.8 + 0.2 * min(signal_strength, 4) / 4)
            
            # Short: Fisher overbought + RSI overbought + HTF not bullish
            if fisher_overbought and rsi_overbought and desired_signal == 0:
                signal_strength = 2
                if price_below_hma_1d:
                    signal_strength += 1
                if price_below_hma_1w:
                    signal_strength += 1
                
                if signal_strength >= 2:
                    desired_signal = -size_short * (0.8 + 0.2 * min(signal_strength, 4) / 4)
        
        # === REGIME 2: TRENDING (CHOP < 38.2) — TREND FOLLOW ===
        elif regime_trend:
            # Long: KAMA bullish + Fisher cross up + volume spike + HTF bullish
            if kama_bullish and fisher_cross_up:
                signal_strength = 1
                if vol_spike:
                    signal_strength += 1
                if price_above_hma_1d:
                    signal_strength += 1
                if price_above_hma_1w:
                    signal_strength += 1
                
                if signal_strength >= 2:
                    desired_signal = size_long * (0.8 + 0.2 * min(signal_strength, 4) / 4)
            
            # Short: KAMA bearish + Fisher cross down + volume spike + HTF bearish
            if kama_bearish and fisher_cross_down and desired_signal == 0:
                signal_strength = 1
                if vol_spike:
                    signal_strength += 1
                if price_below_hma_1d:
                    signal_strength += 1
                if price_below_hma_1w:
                    signal_strength += 1
                
                if signal_strength >= 2:
                    desired_signal = -size_short * (0.8 + 0.2 * min(signal_strength, 4) / 4)
        
        # === REGIME 3: TRANSITION (38.2-61.8) — KAMA TREND ONLY ===
        else:
            if kama_bullish and price_above_hma_1d:
                desired_signal = size_long * 0.7
            elif kama_bearish and price_below_hma_1d:
                desired_signal = -size_short * 0.7
        
        # === CAP SIGNAL TO MAX 0.35 ===
        if desired_signal > 0.35:
            desired_signal = 0.35
        elif desired_signal < -0.35:
            desired_signal = -0.35
        
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
        
        # === FISHER EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and fisher[i] > 1.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (kama_bullish or price_above_hma_1d):
                desired_signal = size_long
            elif position_side < 0 and (kama_bearish or price_below_hma_1d):
                desired_signal = -size_short
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.30
                elif desired_signal >= 0.22:
                    desired_signal = 0.25
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.30
                elif desired_signal <= -0.22:
                    desired_signal = -0.25
                else:
                    desired_signal = -0.15
        
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