#!/usr/bin/env python3
"""
Experiment #021: Bollinger Squeeze + Volume + Choppiness (4h)

HYPOTHESIS: BB Squeeze (low volatility) + volume confirmation + chop regime
- BB squeeze detects consolidation before explosive breakouts (different from Donchian)
- Choppiness determines if we trend-follow or skip (chop>61.8=range, skip)
- Volume confirms the breakout is institutional, not noise
- 1d SMA200 for trend direction filter

WHY IT SHOULD WORK: Low volatility → high volatility transitions are predictable.
When BB width is at low percentile AND volume spikes, the subsequent move is often 3-5 ATR.
This is well-documented phenomenon, different from simple breakout.

TARGET: 150-300 total trades over 4 years (37-75/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(bb_width, period=14):
    """Choppiness Index: 100 * log10(sum ATR) / log10(highest_range)"""
    n = len(bb_width)
    chop = np.full(n, 50.0)  # neutral default
    
    for i in range(period, n):
        if np.isnan(bb_width[i]) or bb_width[i] <= 0:
            continue
        
        # Use sum of BB widths as proxy for range sum
        recent_sum = np.sum(bb_width[max(0, i-period):i+1])
        if recent_sum <= 0:
            continue
        
        # Sum of current BB width as "ATR"
        atr_sum = recent_sum
        
        # Highest-lowest range (proxy with rolling max-min of close)
        # Simple proxy: ATR * period gives range estimate
        if i >= period:
            atr_proxy = np.mean(bb_width[max(0, i-period):i+1]) * period
            if atr_proxy > 0:
                chop[i] = 100 * np.log10(atr_sum / atr_proxy) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d SMA200
    sma200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma200_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    # === 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Bollinger Bands (20 period, 2 std)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = bb_upper - bb_lower
    
    # BB width percentile (detect squeeze)
    bb_width_ma = pd.Series(bb_width).rolling(window=100, min_periods=100).mean().values
    bb_width_std = pd.Series(bb_width).rolling(window=100, min_periods=100).std().values
    bb_zscore = np.where(bb_width_std > 0, (bb_width - bb_width_ma) / bb_width_std, 0)
    
    # Choppiness (simplified)
    chop = np.full(n, 50.0)
    for i in range(14, n):
        if np.isnan(bb_width[i]) or bb_width[i] <= 0:
            continue
        recent_sum = np.nansum(bb_width[max(0, i-14):i+1])
        if recent_sum > 0 and not np.isnan(bb_width_ma[i]):
            atr_sum = np.mean(bb_width[max(0, i-14):i+1]) * 14
            if atr_sum > 0:
                chop[i] = 100 * np.log10(recent_sum / atr_sum) / np.log10(14)
    
    # Volume ratio
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, 1)
    
    # Donchian for exit (20 period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=1).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=1).min().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.28
    
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 250  # BB(20) + SMA200(1d) alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width[i]) or bb_width[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma200_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === Regime check ===
        chop_val = chop[i]
        in_chop = chop_val > 61.8  # Range market, skip
        in_trend = chop_val < 38.2  # Trending, take signals
        
        # === 1d trend filter ===
        htf_bull = close[i] > sma200_aligned[i]
        htf_bear = close[i] < sma200_aligned[i]
        
        # === Squeeze detection ===
        # BB width at low percentile (zscore < -1.0) = squeeze
        is_squeeze = bb_zscore[i] < -0.8
        
        # Volume spike
        vol_spike = vol_ratio[i] > 1.5
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # 1. Squeeze (low volatility)
            # 2. Volume spike (institutional)
            # 3. Price above SMA200 (trend up)
            # 4. In trending chop regime OR breakout from squeeze
            long_squeeze = is_squeeze and vol_spike and htf_bull
            long_breakout = not is_squeeze and vol_spike and htf_bull and (in_trend or chop_val < 55)
            
            if long_squeeze or long_breakout:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # 1. Squeeze (low volatility)
            # 2. Volume spike (institutional)
            # 3. Price below SMA200 (trend down)
            # 4. In trending chop regime OR breakout from squeeze
            short_squeeze = is_squeeze and vol_spike and htf_bear
            short_breakout = not is_squeeze and vol_spike and htf_bear and (in_trend or chop_val < 55)
            
            if short_squeeze or short_breakout:
                desired_signal = -SIZE
        
        # === STOPLOSS: 2.5 ATR ===
        if in_position:
            if position_side > 0:
                stop = entry_price - 2.5 * entry_atr
                if low[i] < stop:
                    desired_signal = 0.0
                # Donchian trailing exit
                if low[i] < donchian_low[i]:
                    desired_signal = 0.0
            elif position_side < 0:
                stop = entry_price + 2.5 * entry_atr
                if high[i] > stop:
                    desired_signal = 0.0
                # Donchian trailing exit
                if high[i] > donchian_high[i]:
                    desired_signal = 0.0
        
        # === MIN HOLD: 3 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals