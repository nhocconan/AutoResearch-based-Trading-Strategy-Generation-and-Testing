#!/usr/bin/env python3
"""
Experiment #6399: 6h Donchian(20) breakout + 12h volume confirmation + 1d ADX(14) regime filter
HYPOTHESIS: 6h Donchian breakouts with volume confirmation (>1.5x 12h average) and 1d ADX(14)>25 (trending regime) capture strong momentum while avoiding chop. The 1d ADX filter ensures we only trade in trending markets, reducing whipsaws in ranging conditions. Volume confirmation filters low-momentum breakouts. Discrete sizing (0.25) balances profit potential and drawdown control. Target: 75-150 trades over 4 years. Works in bull via breakouts with ADX uptrend, in bear via short breakdowns with ADX downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6399_6h_donchian20_12h_vol_1d_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 12h data for volume average ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        vol_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
    else:
        vol_12h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for ADX(14) regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        # Calculate ADX components
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
        tr_1d[0] = tr1[0]
        
        # Directional Movement
        up_move = high_1d - np.roll(high_1d, 1)
        down_move = np.roll(low_1d, 1) - low_1d
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
        plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
        minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
        
        # Directional Indicators
        plus_di_14 = 100 * plus_dm_14 / np.where(tr_14 > 0, tr_14, 1)
        minus_di_14 = 100 * minus_dm_14 / np.where(tr_14 > 0, tr_14, 1)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di_14 - minus_di_14) / np.where((plus_di_14 + minus_di_14) > 0, (plus_di_14 + minus_di_14), 1)
        adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 14) + 1  # Donchian, volume avg, ATR, ADX lookback + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_12h_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Donchian low (failed breakout)
                # 3. ADX drops below 20 (regime change to ranging)
                if price <= stop_price or price <= donchian_low[i] or adx_1d_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Donchian high (failed breakout)
                # 3. ADX drops below 20 (regime change to ranging)
                if price >= stop_price or price >= donchian_high[i] or adx_1d_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume[i] > 1.5 * vol_12h_aligned[i]  # Volume filter
        trending_regime = adx_1d_aligned[i] > 25  # Only trade in trending markets
        
        # Entry logic based on ADX regime:
        # Long: breakout up + volume + trending regime
        # Short: breakout down + volume + trending regime
        
        long_entry = breakout_up and volume_confirmed and trending_regime
        short_entry = breakout_down and volume_confirmed and trending_regime
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals