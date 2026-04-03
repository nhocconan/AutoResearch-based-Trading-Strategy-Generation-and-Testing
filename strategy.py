#!/usr/bin/env python3
"""
Experiment #295: 6h Elder Ray + 1d ADX Regime + Volume Confirmation
HYPOTHESIS: Elder Ray (Bull/Bear Power) identifies momentum strength while 1d ADX regime filters for trending vs ranging markets. In trending markets (ADX>25), follow Elder Ray signals. In ranging markets (ADX<20), fade extreme Bull/Bear Power readings. Volume confirmation reduces false signals. Designed to work in both bull (trend following) and bear (mean reversion in ranges) markets. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_295_6h_elder_ray_1d_adx_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    period = 14
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=period, min_periods=period).mean()
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # === 6h Indicators: Elder Ray (Bull Power / Bear Power) ===
    # EMA(13) as trend proxy
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema_13
    # Bear Power = Low - EMA(13)
    bear_power = low - ema_13
    
    # Normalize by ATR(14) for consistency across volatility regimes
    # ATR(14) on 6h
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_6h = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    bull_power_norm = bull_power / atr_6h
    bear_power_norm = bear_power / atr_6h
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(60, 20)  # ATR/EMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power_norm[i]) or 
            np.isnan(bear_power_norm[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(ema_13[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Regime Detection ---
        adx_val = adx_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # --- Elder Ray Signals ---
        bullish_momentum = bull_power_norm[i] > 0.5  # Strong bullish pressure
        bearish_momentum = bear_power_norm[i] < -0.5  # Strong bearish pressure
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2*ATR below entry
                stop_level = entry_price - 2.0 * atr_6h[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit: momentum fades
                if not bullish_momentum and bars_since_entry >= 3:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2*ATR above entry
                stop_level = entry_price + 2.0 * atr_6h[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit: momentum fades
                if not bearish_momentum and bars_since_entry >= 3:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            if is_trending:
                # Trending market: follow momentum
                if bullish_momentum:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif bearish_momentum:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            elif is_ranging:
                # Ranging market: fade extreme momentum
                if bullish_momentum:
                    # Extreme bullish power in range = short opportunity
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                elif bearish_momentum:
                    # Extreme bearish power in range = long opportunity
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            else:
                # Transition regime (ADX 20-25): no clear edge
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals