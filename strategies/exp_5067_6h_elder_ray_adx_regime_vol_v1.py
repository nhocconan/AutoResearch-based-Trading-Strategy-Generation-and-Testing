#!/usr/bin/env python3
"""
Experiment #5067: 6h Elder Ray Power + ADX Regime + Volume Spike
HYPOTHESIS: On 6h timeframe, Elder Ray Bull/Bear Power combined with ADX regime filter (ADX>25 for trending, ADX<20 for ranging) captures institutional momentum with controlled frequency. Volume > 1.5x average confirms participation. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag. Works in bull markets (buy when Bull Power>0, ADX>25) and bear markets (sell when Bear Power<0, ADX>25). Uses 1d HTF for weekly trend filter (price > weekly EMA50 for long bias, < for short bias) to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5067_6h_elder_ray_adx_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly EMA50 for trend bias ===
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        weekly_ema50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        weekly_ema50_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Elder Ray Power (Bull/Bear) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h Indicators: ADX(14) for regime filter ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus14 = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus14 = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(13, 14, 20, 50)  # EMA13, ADX, Vol MA, Weekly EMA50 warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ratio[i]) or np.isnan(weekly_ema50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse on opposite signal ---
        if in_position:
            # Long exit: Bull Power turns negative OR ADX weakens (<20) OR price crosses below weekly EMA50
            if position_side > 0:
                if (bull_power[i] <= 0) or (adx[i] < 20) or (price < weekly_ema50_aligned[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            # Short exit: Bear Power turns positive OR ADX weakens (<20) OR price crosses above weekly EMA50
            else:
                if (bear_power[i] >= 0) or (adx[i] < 20) or (price > weekly_ema50_aligned[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Regime filter: ADX > 25 for trending market
        trending = adx[i] > 25
        
        # Weekly trend bias from 1d HTF
        weekly_bullish = price > weekly_ema50_aligned[i]
        weekly_bearish = price < weekly_ema50_aligned[i]
        
        # Long: Bull Power positive + trending + weekly bullish bias + volume
        # Short: Bear Power negative + trending + weekly bearish bias + volume
        long_signal = (bull_power[i] > 0) and trending and weekly_bullish and vol_confirm
        short_signal = (bear_power[i] < 0) and trending and weekly_bearish and vol_confirm
        
        if long_signal:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_signal:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals