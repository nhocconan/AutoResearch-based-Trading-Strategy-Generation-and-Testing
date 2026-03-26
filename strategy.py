#!/usr/bin/env python3
"""
Experiment #027: 4h BB Squeeze Expansion + ADX Trend + Volume Spike

HYPOTHESIS: Bollinger Band squeeze (width at 30-bar low) marks low-volatility 
compression before explosive moves. Combined with ADX trend confirmation 
(direction) and volume spike (institutional participation), this catches 
breakouts while avoiding choppy range-bound markets.

TIMEFRAME: 4h primary
HTF: 1d for trend bias (SMA200)
TARGET: 75-200 total trades over 4 years (19-50/year)
HARD MAX: 400 total

WHY IT WORKS:
- BB squeeze is a proven volatility signal (Bollinger's original work)
- ADX confirms trend is real (avoids whipsaws in ranging markets)
- Volume spike confirms institutional participation
- Works in both bull (long squeeze expansions) and bear (short squeeze expansions)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_adx_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's method (equivalent to EMA with alpha=1/period)
    atr_smooth = np.zeros(n)
    atr_smooth[period-1] = np.sum(tr[0:period])
    for i in range(period, n):
        atr_smooth[i] = atr_smooth[i-1] - atr_smooth[i-1]/period + tr[i]
    
    plus_di_smooth = np.zeros(n)
    plus_di_smooth[period-1] = np.sum(plus_dm[0:period])
    for i in range(period, n):
        plus_di_smooth[i] = plus_di_smooth[i-1] - plus_di_smooth[i-1]/period + plus_dm[i]
    
    minus_di_smooth = np.zeros(n)
    minus_di_smooth[period-1] = np.sum(minus_dm[0:period])
    for i in range(period, n):
        minus_di_smooth[i] = minus_di_smooth[i-1] - minus_di_smooth[i-1]/period + minus_dm[i]
    
    # ADX
    adx = np.full(n, np.nan)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_di_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_di_smooth[i] / atr_smooth[i]
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
                if i == period:
                    adx[i] = dx
                else:
                    adx[i] = (adx[i-1] * (period - 1) + dx) / period
    
    return adx, plus_di, minus_di

def calculate_bollinger_bands(close, period=20, num_std=2.0):
    """Bollinger Bands"""
    n = len(close)
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mid + num_std * std
    lower = mid - num_std * std
    
    return upper, mid, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA for trend bias
    sma_200_1d = df_1d['close'].rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ADX for trend strength
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, num_std=2.0)
    
    # BB Width (squeeze detection)
    bb_width = bb_upper - bb_lower
    bb_width_ma = pd.Series(bb_width).rolling(window=30, min_periods=30).mean().values
    bb_width_percentile = np.zeros(n)
    for i in range(29, n):
        if not np.isnan(bb_width[i]):
            window = bb_width[max(0, i-30):i+1]
            window = window[~np.isnan(window)]
            if len(window) >= 15:
                bb_width_percentile[i] = (bb_width[i] - np.min(window)) / (np.max(window) - np.min(window) + 1e-10)
    
    # Volume analysis
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Price momentum (rate of change)
    roc = pd.Series(close).pct_change(periods=8).values * 100
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx[i]) or np.isnan(bb_width[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SQUEEZE DETECTION ===
        # Squeeze: BB width at low percentile (compression before expansion)
        is_squeeze = bb_width_percentile[i] < 0.20
        
        # === EXPANSION DETECTION ===
        # Expansion: BB width expanding from squeeze low
        was_squeeze_recently = np.any(bb_width_percentile[max(0, i-8):i] < 0.25)
        is_expanding = bb_width[i] > bb_width_ma[i] * 1.1 if not np.isnan(bb_width_ma[i]) else False
        
        # === TREND CONFIRMATION (ADX) ===
        adx_val = adx[i]
        trend_strength = adx_val > 22  # ADX above 22 = trending
        
        # === DIRECTION (DI crossover) ===
        plus_di_val = plus_di[i] if not np.isnan(plus_di[i]) else 0
        minus_di_val = minus_di[i] if not np.isnan(minus_di[i]) else 0
        bullish_dmi = plus_di_val > minus_di_val
        bearish_dmi = minus_di_val > plus_di_val
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === TREND BIAS (1d SMA200) ===
        price_above_200 = close[i] > sma_200_aligned[i]
        
        # === MOMENTUM ===
        roc_val = roc[i] if not np.isnan(roc[i]) else 0
        positive_momentum = roc_val > -1.0
        negative_momentum = roc_val < 1.0
        
        # === ENTRY LOGIC ===
        # Key insight: squeeze + expansion + trend + volume = high probability move
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Need: squeeze was active, now expanding, bullish DMI, volume spike, above SMA200
            if is_expanding and was_squeeze_recently and trend_strength:
                if bullish_dmi and vol_spike and price_above_200:
                    desired_signal = SIZE
                # Also allow if ADX is strong even without clear DMI
                elif adx_val > 30 and positive_momentum and price_above_200:
                    desired_signal = SIZE * 0.5  # Half size for weaker signal
            
            # === SHORT ENTRY ===
            # Need: squeeze was active, now expanding, bearish DMI, volume spike, below SMA200
            if is_expanding and was_squeeze_recently and trend_strength:
                if bearish_dmi and vol_spike and not price_above_200:
                    desired_signal = -SIZE
                # Also allow if ADX is strong even without clear DMI
                elif adx_val > 30 and negative_momentum and not price_above_200:
                    desired_signal = -SIZE * 0.5  # Half size for weaker signal
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT: Opposite signal or trend exhaustion ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: bearish DMI crossover OR ADX drops OR price closes below SMA200
            if bearish_dmi and plus_di_val < minus_di_val - 5:
                exit_triggered = True
            if adx_val < 18:
                exit_triggered = True
            if close[i] < sma_200_aligned[i] * 0.98:  # 2% buffer
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: bullish DMI crossover OR ADX drops OR price closes above SMA200
            if bullish_dmi and plus_di_val > minus_di_val + 5:
                exit_triggered = True
            if adx_val < 18:
                exit_triggered = True
            if close[i] > sma_200_aligned[i] * 1.02:  # 2% buffer
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals