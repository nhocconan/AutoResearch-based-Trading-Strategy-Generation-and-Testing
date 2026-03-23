#!/usr/bin/env python3
"""
Experiment #1256: 12h Primary + 1d HTF — KAMA + Fisher Transform + Volume

Hypothesis: Recent failures all have Sharpe=0.000 = ZERO TRADES due to overly strict
entry conditions. This strategy uses:
1. KAMA (Kaufman Adaptive MA) - adapts to market noise, proven better than HMA/EMA
2. Fisher Transform - catches reversals in bear markets (research shows 70%+ win rate)
3. Volume confirmation - but LOOSE threshold (0.8x avg, not 1.5x)
4. Asymmetric entries - long easier in bull, short easier in bear
5. Lower ADX threshold (12) and wider Fisher bands (-1.8/+1.8)

Key differences from failed #1246, #1247, #1252, #1253:
- Fisher Transform instead of CRSI (better for bear market reversals)
- KAMA instead of HMA (adaptive to volatility)
- Much looser entry thresholds to ensure >=30 trades/train
- No hysteresis buffer blocking signals
- Volume filter is permissive (0.8x not 1.5x)

Target: Sharpe > 0.612, trades >= 40 train, >= 5 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_volume_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # KAMA calculation
    kama[period - 1] = close[period - 1]
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """Ehlers Fisher Transform - catches reversals
    Long when Fisher crosses above -1.5
    Short when Fisher crosses below +1.5
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    for i in range(period - 1, n):
        # Price position within range
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh > ll:
            price_pos = (close[i] - ll) / (hh - ll)
            # Clamp to avoid division issues
            price_pos = max(0.001, min(0.999, price_pos))
            
            # Fisher calculation
            fisher_val = 0.5 * np.log((1 + price_pos) / (1 - price_pos))
            
            if i >= period:
                fisher_signal[i] = 0.67 * fisher_val + 0.33 * fisher_signal[i - 1]
            else:
                fisher_signal[i] = fisher_val
            
            fisher[i] = fisher_val
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    sma = np.full(n, np.nan)
    
    if n < period:
        return sma
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def calculate_volume_ma(volume, period=20):
    """Volume Moving Average"""
    n = len(volume)
    vol_ma = np.full(n, np.nan)
    
    if n < period:
        return vol_ma
    
    for i in range(period - 1, n):
        vol_ma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for macro trend filter
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=14)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (12h) indicators
    kama_fast = calculate_kama(close, period=8, fast_period=2, slow_period=20)
    kama_slow = calculate_kama(close, period=21, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crossovers
    prev_fisher = np.nan
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(adx[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(sma_200[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1d KAMA) ===
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === LOCAL TREND (12h KAMA) ===
        kama_bull = kama_fast[i] > kama_slow[i]
        kama_bear = kama_fast[i] < kama_slow[i]
        
        # === TREND STRENGTH (ADX) - LOOSE threshold ===
        trend_strong = adx[i] > 12.0
        
        # === VOLUME FILTER - PERMISSIVE ===
        volume_ok = volume[i] > 0.7 * vol_ma[i]
        
        # === SMA 200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Fisher crossover detection
        fisher_cross_up = False
        fisher_cross_down = False
        
        if not np.isnan(prev_fisher) and not np.isnan(fisher[i]):
            if prev_fisher < -1.5 and fisher[i] >= -1.5:
                fisher_cross_up = True
            if prev_fisher > 1.5 and fisher[i] <= 1.5:
                fisher_cross_down = True
        
        prev_fisher = fisher[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG entries (easier in bull regime)
        if macro_bull or above_sma200:
            # Trend follow: KAMA bull + ADX strong + volume
            if kama_bull and trend_strong and volume_ok:
                desired_signal = BASE_SIZE
            # Mean revert: Fisher oversold + KAMA not strongly bear
            elif fisher_oversold and not kama_bear:
                desired_signal = BASE_SIZE
            # Fisher crossover up
            elif fisher_cross_up and volume_ok:
                desired_signal = BASE_SIZE
        
        # SHORT entries (easier in bear regime)
        elif macro_bear or not above_sma200:
            # Trend follow: KAMA bear + ADX strong + volume
            if kama_bear and trend_strong and volume_ok:
                desired_signal = -BASE_SIZE
            # Mean revert: Fisher overbought + KAMA not strongly bull
            elif fisher_overbought and not kama_bull:
                desired_signal = -BASE_SIZE
            # Fisher crossover down
            elif fisher_cross_down and volume_ok:
                desired_signal = -BASE_SIZE
        
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
        
        # === OUTPUT SIGNAL ===
        final_signal = desired_signal
        
        # === DISCRETIZE SIGNAL VALUES ===
        if final_signal > 0.1:
            final_signal = BASE_SIZE
        elif final_signal < -0.1:
            final_signal = -BASE_SIZE
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