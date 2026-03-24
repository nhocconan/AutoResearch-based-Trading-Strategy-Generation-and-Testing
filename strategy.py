#!/usr/bin/env python3
"""
Experiment #207: 6h Primary + 1d HTF — KAMA Adaptive Trend + Fisher Reversals

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). This strategy
combines Kaufman Adaptive Moving Average (KAMA) for volatility-adaptive trend
following with Ehlers Fisher Transform for precise reversal entries.

Key innovations:
1. KAMA (ER=10): Adapts to market efficiency - fast in trends, slow in chop
2. Fisher Transform (period=9): Normalizes price to Gaussian distribution for
   clear reversal signals at ±1.5 extremes
3. ADX(14) regime filter: >25 = trend follow, <20 = mean revert
4. 1d HMA(50) for major trend bias - only trade in HTF direction
5. ATR ratio (ATR7/ATR30) for vol spike detection - enter on vol crush

Entry logic:
- TREND regime (ADX>25): KAMA cross + Fisher confirmation + 1d trend align
- MEAN REVERT regime (ADX<20): Fisher extreme (<-1.5 or >+1.5) + 1d trend align
- Vol spike entry: ATR7/ATR30 > 2.0 + Fisher extreme = reversal play

Position sizing: 0.25 base, 0.30 strong signals (discrete levels)
Stoploss: 2.5x ATR trailing stop

Target: Sharpe>0.40 (beat current 6h best), 30-60 trades/year, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_fisher_adx_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for clear reversal signals
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - LL) / (HH - LL) - 0.33
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            X = 0.67 * ((close[i] - lowest) / price_range - 0.33)
            X = np.clip(X, -0.999, 0.999)  # Prevent division by zero
            fisher[i] = 0.5 * np.log((1.0 + X) / (1.0 - X))
            
            if i > period - 1:
                fisher_signal[i] = fisher[i - 1]
        else:
            fisher[i] = 0.0
            if i > period - 1:
                fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed DM and TR
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 1e-10:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is smoothed DX
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period * 2 - 1:] = adx_raw[period * 2 - 1:]
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr7 = calculate_atr(high, low, close, period=7)
    atr30 = calculate_atr(high, low, close, period=30)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (ADX) ===
        is_trending = adx[i] > 25.0
        is_ranging = adx[i] < 20.0
        
        # === KAMA TREND ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_bull_cross = fisher[i] > fisher_signal[i] and fisher_signal[i] < -1.0
        fisher_bear_cross = fisher[i] < fisher_signal[i] and fisher_signal[i] > 1.0
        
        # === VOL SPIKE DETECTION ===
        vol_spike = False
        if not np.isnan(atr7[i]) and not np.isnan(atr30[i]) and atr30[i] > 1e-10:
            vol_spike = (atr7[i] / atr30[i]) > 2.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (ADX > 25) - Follow KAMA trend with Fisher confirmation
        if is_trending:
            # Long: KAMA bull + HTF bull + Fisher not overbought
            if kama_bull and htf_bull and not fisher_overbought:
                desired_signal = SIZE_BASE
            
            # Strong long: Add Fisher bull cross
            if kama_bull and htf_bull and fisher_bull_cross:
                desired_signal = SIZE_STRONG
            
            # Short: KAMA bear + HTF bear + Fisher not oversold
            if kama_bear and htf_bear and not fisher_oversold:
                desired_signal = -SIZE_BASE
            
            # Strong short: Add Fisher bear cross
            if kama_bear and htf_bear and fisher_bear_cross:
                desired_signal = -SIZE_STRONG
        
        # REGIME 2: RANGING (ADX < 20) - Mean revert with Fisher extremes
        elif is_ranging:
            # Long: Fisher oversold + HTF not strongly bear
            if fisher_oversold and (htf_bull or adx[i] < 15):
                desired_signal = SIZE_BASE
            
            # Strong long: Add vol spike (panic reversal)
            if fisher_oversold and vol_spike:
                desired_signal = SIZE_STRONG
            
            # Short: Fisher overbought + HTF not strongly bull
            if fisher_overbought and (htf_bear or adx[i] < 15):
                desired_signal = -SIZE_BASE
            
            # Strong short: Add vol spike
            if fisher_overbought and vol_spike:
                desired_signal = -SIZE_STRONG
        
        # REGIME 3: TRANSITION (ADX 20-25) - Reduced size, require stronger confirmation
        else:
            # Only enter on Fisher cross + HTF alignment
            if fisher_bull_cross and htf_bull:
                desired_signal = SIZE_BASE * 0.6
            elif fisher_bear_cross and htf_bear:
                desired_signal = -SIZE_BASE * 0.6
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.6
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals