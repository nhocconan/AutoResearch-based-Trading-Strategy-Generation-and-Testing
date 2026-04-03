#!/usr/bin/env python3
"""
Experiment #1874: 1h HTF Donchian Breakout + Volume + Session Filter
HYPOTHESIS: Use 4h/1d Donchian channels for trend direction and structure, 
1h for precise entry timing with volume confirmation and session filter (08-20 UTC). 
Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing 0.20.
Uses price channels (Donchian) which work in both bull/bear markets by capturing 
breakouts from consolidation. Volume confirms institutional participation. 
Session filter reduces noise during low-liquidity hours.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1874_1h_donchian_htf_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for Donchian channels (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20) - upper/lower bands
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe (with shift(1) for completed bars only)
    donchian_upper_4h = align_htf_to_ltf(prices, df_4h, high_20)
    donchian_lower_4h = align_htf_to_ltf(prices, df_4h, low_20)
    
    # === HTF: 1d EMA(50) for higher timeframe trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Session filter: 08-20 UTC (pre-compute for efficiency) ===
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for Donchian(20) and EMA(50)
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss or HTF trend reversal ---
        if in_position:
            # Stoploss: 2 * ATR(14) approximation using 20-bar range
            lookback = min(20, i+1)
            if lookback >= 2:
                period_high = np.max(high[i-lookback+1:i+1])
                period_low = np.min(low[i-lookback+1:i+1])
                approx_atr = (period_high - period_low) / 2  # rough ATR estimate
            else:
                approx_atr = price * 0.01  # fallback 1%
            
            stoploss_hit = False
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * approx_atr:
                    stoploss_hit = True
            else:  # Short
                if price > entry_price + 2.0 * approx_atr:
                    stoploss_hit = True
            
            # Exit if 1d trend flips
            trend_reversal = (position_side > 0 and trend_1d_aligned[i] < 0) or \
                           (position_side < 0 and trend_1d_aligned[i] > 0)
            
            if stoploss_hit or trend_reversal:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long: price breaks above 4h Donchian upper band
            if trend_bias > 0 and price > donchian_upper_4h[i]:
                in_position = True
                position_side = 1
                entry_price = price
                signals[i] = SIZE
            # Short: price breaks below 4h Donchian lower band
            elif trend_bias < 0 and price < donchian_lower_4h[i]:
                in_position = True
                position_side = -1
                entry_price = price
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals