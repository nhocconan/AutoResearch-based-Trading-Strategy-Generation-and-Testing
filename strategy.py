#!/usr/bin/env python3
"""
Experiment #011: 4h Primary + 1d HTF — Fisher Transform + Keltner Squeeze + ADX Regime

Hypothesis: The current best (CRSI + Choppiness) works but can be improved by:
1. Fisher Transform provides cleaner reversal signals than RSI/CRSI (less whipsaw)
2. Keltner Channel squeeze detection identifies low-volatility breakouts
3. ADX confirms trend strength before entering trend trades
4. Volatility-adjusted sizing reduces exposure during high-vol periods
5. Tighter 2.0x ATR stoploss reduces drawdown vs 2.5x

Key differences from exp#001:
- Fisher Transform instead of CRSI (proven in literature for reversals)
- Keltner Channel squeeze instead of Choppiness Index
- ADX for trend strength filter
- Volatility-adjusted position sizing
- 2.0x ATR stoploss (tighter risk management)

Entry Logic:
- MEAN REVERT: Fisher < -1.5 + price outside Keltner lower → long
- TREND: ADX > 25 + Fisher cross + 1d HMA confirmation → follow trend
- Size: 0.25-0.30 base, reduced to 0.20 in high vol or against HTF

Target: Sharpe > 0.15, trades > 30/symbol train, DD > -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_keltner_adx_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for cleaner reversal signals
    Long: Fisher crosses above -1.5 from below
    Short: Fisher crosses below +1.5 from above
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Calculate median price
    median_price = np.zeros(n)
    for i in range(n):
        median_price[i] = (high[i] + low[i]) / 2.0
    
    # Normalize price over period
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        # Normalize to -1 to +1 range
        normalized = 2.0 * (median_price[i] - lowest) / (highest - lowest) - 1.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > 0:
            fisher_prev[i] = fisher[i - 1]
    
    return fisher, fisher_prev

def calculate_keltner_channels(high, low, close, ema_period=20, atr_period=10, multiplier=2.0):
    """
    Keltner Channels
    Middle: EMA(20)
    Upper: EMA(20) + multiplier * ATR(10)
    Lower: EMA(20) - multiplier * ATR(10)
    Squeeze: when price breaks outside channels
    """
    n = len(close)
    if n < ema_period + atr_period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # EMA for middle line
    middle = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    
    # ATR calculation
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    
    return upper, middle, lower

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    ADX > 25 = trending market
    ADX < 20 = ranging market
    """
    n = len(close)
    if n < period * 2 + 5:
        return np.full(n, np.nan)
    
    # Calculate DM and TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
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
    
    # Smooth DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = SMA of DX
    adx = np.full(n, np.nan)
    for i in range(period * 2, n):
        adx[i] = np.mean(dx[i - period + 1:i + 1])
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_volatility_ratio(atr_short, atr_long):
    """ATR ratio for volatility regime detection"""
    n = len(atr_short)
    ratio = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr_short[i]) and not np.isnan(atr_long[i]) and atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    keltner_upper, keltner_middle, keltner_lower = calculate_keltner_channels(high, low, close, ema_period=20, atr_period=10, multiplier=2.0)
    adx = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    vol_ratio = calculate_volatility_ratio(atr_14, atr_30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    MAX_SIZE = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        if np.isnan(fisher[i]) or np.isnan(keltner_middle[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        is_trending = adx[i] > 25.0
        is_ranging = adx[i] < 20.0
        
        # === VOLATILITY REGIME ===
        is_high_vol = not np.isnan(vol_ratio[i]) and vol_ratio[i] > 1.5
        is_low_vol = not np.isnan(vol_ratio[i]) and vol_ratio[i] < 0.8
        
        # === HTF TREND BIAS ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = BASE_SIZE
        
        # Reduce size in high volatility
        if is_high_vol:
            signal_strength = REDUCED_SIZE
        
        if is_ranging or is_low_vol:
            # MEAN REVERSION REGIME - Fisher extremes
            # Long: Fisher < -1.5 and crossing up + price below Keltner lower
            if fisher[i] < -1.5 and fisher_prev[i] < fisher[i] and close[i] < keltner_lower[i]:
                if hma_1d_bull:
                    desired_signal = signal_strength
                else:
                    desired_signal = REDUCED_SIZE
            
            # Short: Fisher > 1.5 and crossing down + price above Keltner upper
            elif fisher[i] > 1.5 and fisher_prev[i] > fisher[i] and close[i] > keltner_upper[i]:
                if hma_1d_bear:
                    desired_signal = -signal_strength
                else:
                    desired_signal = -REDUCED_SIZE
        
        elif is_trending:
            # TREND REGIME - follow HTF bias with Fisher confirmation
            # Long: 1d bullish + Fisher crossing above -1.5
            if hma_1d_bull and fisher[i] > -1.5 and fisher_prev[i] <= -1.5:
                desired_signal = signal_strength
            
            # Short: 1d bearish + Fisher crossing below 1.5
            elif hma_1d_bear and fisher[i] < 1.5 and fisher_prev[i] >= 1.5:
                desired_signal = -signal_strength
        
        else:
            # NEUTRAL ADX (20-25) - only trade with strong HTF confirmation
            if hma_1d_bull and fisher[i] > 0:
                desired_signal = REDUCED_SIZE
            elif hma_1d_bear and fisher[i] < 0:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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