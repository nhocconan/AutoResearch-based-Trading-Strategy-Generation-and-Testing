#!/usr/bin/env python3
"""
Experiment #009: 4h BB Squeeze + Donchian Breakout + Choppiness Regime

HYPOTHESIS: Combine volatility compression detection (BB squeeze) with price 
channel breakout and regime filter. BB squeeze identifies low volatility periods 
where momentum is building; breakout captures the release.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Squeeze forms → release above upper band → strong momentum continuation
- Bear: Squeeze forms → breakdown below lower band → strong short continuation
- Choppiness filter prevents whipsaws in ranging markets (skip when CHOP > 61.8)

CORE CONDITIONS (3 total):
1. BB Squeeze: width percentile < 20 (volatility compression)
2. Donchian(20) breakout: high breaks prior 20-bar high (momentum confirmation)
3. Choppiness regime: CHOP < 50 (trending environment)

EXPECTED TRADES: 100-200 total over 4 years (25-50/year)
- Squeeze forms ~monthly; breakout confirmation reduces by ~50%
- Choppiness filter skips ~30% of squeeze setups in range markets
- Final: ~120-180 total over 4 years
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_donchian_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def choppiness_index(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = ranging"""
    n = len(close)
    ci = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = high[i-period:i+1].max()
        lowest = low[i-period:i+1].min()
        range_sum = highest - lowest
        
        if range_sum > 0:
            atr_sum = 0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
                atr_sum += tr
            
            ci[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return ci

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA50 for trend direction
    daily_ema50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_width = bb_upper - bb_lower
    
    # BB Width percentile (50 bars) - squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_ma50 = bb_width_series.rolling(window=50, min_periods=20).mean().values
    bb_width_std50 = bb_width_series.rolling(window=50, min_periods=20).std().values
    bb_width_z = (bb_width - bb_width_ma50) / np.where(bb_width_std50 > 0, bb_width_std50, 1)
    
    # Donchian Channel (20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, 1)
    
    # Choppiness Index (14)
    ci = choppiness_index(high, low, close, period=14)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # Donchian20, BB20, BW50, ATR14, EMA50
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_z[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME: Choppiness < 50 = trending (use breakout), > 61.8 = ranging (skip) ===
        trending = not np.isnan(ci[i]) and ci[i] < 50.0
        ranging = not np.isnan(ci[i]) and ci[i] > 61.8
        
        if ranging:
            # Skip entries in ranging markets
            if in_position:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION: Daily EMA50 ===
        bull_trend = close[i] > ema50_aligned[i]
        bear_trend = close[i] < ema50_aligned[i]
        
        # === BB SQUEEZE: Width z-score < 0 (below average = compression) ===
        squeeze = bb_width_z[i] < 0.0
        
        # === VOLUME CONFIRMATION: > 1.2x average ===
        vol_confirm = vol_ratio[i] > 1.2
        
        # === DONCHIAN BREAKOUT ===
        # Long: close or high breaks above prior 20-bar high
        # Short: close or low breaks below prior 20-bar low
        prev_high_20 = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_low_20 = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_high_20) and 
                           high[i] >= prev_high_20)
        bearish_breakout = (not np.isnan(prev_low_20) and 
                           low[i] <= prev_low_20)
        
        # Minimum hold: 4 bars (reduce fee churn)
        min_hold_passed = (i - entry_bar) >= 4 if in_position else True
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            stop_price = entry_price - 2.5 * entry_atr if position_side > 0 else entry_price + 2.5 * entry_atr
            stop_hit = (position_side > 0 and low[i] < stop_price) or (position_side < 0 and high[i] > stop_price)
            
            # Trend exit: price crosses EMA50
            trend_exit = (position_side > 0 and close[i] < ema50_aligned[i]) or \
                        (position_side < 0 and close[i] > ema50_aligned[i])
            
            # Choppiness exit: if market turns ranging, exit
            chop_exit = not np.isnan(ci[i]) and ci[i] > 61.8
            
            if stop_hit or (min_hold_passed and (trend_exit or chop_exit)):
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Squeeze + breakout + volume + bull trend + trending market
            if squeeze and bullish_breakout and vol_confirm and bull_trend and trending:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Squeeze + breakdown + volume + bear trend + trending market
            elif squeeze and bearish_breakout and vol_confirm and bear_trend and trending:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
    
    return signals