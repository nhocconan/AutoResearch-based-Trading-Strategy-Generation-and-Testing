#!/usr/bin/env python3
"""
Experiment #154: 1h Strategy with 4h/1d HTF Filters for BTC/ETH/SOL

HYPOTHESIS: Use 4h Donchian breakout direction + 1d EMA200 trend filter for signal direction,
and 1h timeframe only for precise entry timing with volume confirmation. This reduces
overtrading by letting HTF define the regime while 1h captures optimal entry points.
Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
Session filter (08-20 UTC) avoids low-liquidity periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for Donchian direction (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Donchian Channel (20) on 4h
    if len(df_4h) >= 20:
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        donchian_h_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
        donchian_l_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
        donchian_m_4h = (donchian_h_4h + donchian_l_4h) / 2
        
        # Determine 4h trend direction based on price vs Donchian middle
        close_4h = df_4h['close'].values
        trend_4h = np.where(close_4h > donchian_m_4h, 1, 
                           np.where(close_4h < donchian_m_4h, -1, 0))
        
        # Align to 1h timeframe with proper shift(1) for completed bars only
        donchian_h_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_h_4h)
        donchian_l_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_l_4h)
        trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    else:
        donchian_h_4h_aligned = np.full(n, np.nan)
        donchian_l_4h_aligned = np.full(n, np.nan)
        trend_4h_aligned = np.zeros(n)
    
    # === HTF: 1d data for EMA200 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(200) on 1d close
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    else:
        ema_200_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for HTF EMA200
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_h_4h_aligned[i]) or np.isnan(donchian_l_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- HTF Direction Filters ---
        # 4h Donchian breakout direction (long if price above middle, short if below)
        dir_4h = trend_4h_aligned[i]
        
        # 1d EMA200 trend filter
        price_above_1d_ema = close[i] > ema_200_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_200_1d_aligned[i]
        
        # Combined direction signal: must agree on both timeframes
        long_dir = (dir_4h > 0) and price_above_1d_ema
        short_dir = (dir_4h < 0) and price_below_1d_ema
        
        # --- 1h Entry Timing with Volume Confirmation ---
        # Volume spike (> 1.5x average) for entry timing
        volume_spike = vol_ratio[i] > 1.5
        
        # 1h price position relative to 4h Donchian levels
        near_4h_high = close[i] > donchian_h_4h_aligned[i] * 0.999  # Within 0.1% of 4h high
        near_4h_low = close[i] < donchian_l_4h_aligned[i] * 1.001   # Within 0.1% of 4h low
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: 4h/1d bullish alignment + volume spike + near 4h high
        long_entry = long_dir and volume_spike and near_4h_high
        
        # Short: 4h/1d bearish alignment + volume spike + near 4h low
        short_entry = short_dir and volume_spike and near_4h_low
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals