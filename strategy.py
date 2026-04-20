#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Breakout_VolumeTrend_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: 20-period EMA for trend direction ===
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # === 4h: Donchian channel (20-period high/low) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian upper and lower bands
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h: ATR(14) for volatility ===
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h: Volume confirmation (current volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        ema_trend = ema_20_1d_aligned[i]
        dc_up = dc_upper[i]
        dc_low = dc_lower[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_vol = prices['volume'].iloc[i]
        vol_avg = vol_ma[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_trend) or np.isnan(dc_up) or np.isnan(dc_low) or 
            np.isnan(current_atr) or np.isnan(vol_avg)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = current_vol > 1.5 * vol_avg
        
        if position == 0:
            # Long conditions:
            # 1. Price above 1d EMA20 (uptrend)
            # 2. Price breaks above Donchian upper with volume
            if (current_close > ema_trend and
                current_close > dc_up and
                vol_condition):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price below 1d EMA20 (downtrend)
            # 2. Price breaks below Donchian lower with volume
            elif (current_close < ema_trend and
                  current_close < dc_low and
                  vol_condition):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below 1d EMA20 (trend change)
            # 2. Price hits Donchian lower (mean reversion)
            # 3. ATR-based stop loss
            if (current_close < ema_trend or
                current_close <= dc_low or
                current_close < entry_price - 2.0 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above 1d EMA20 (trend change)
            # 2. Price hits Donchian upper (mean reversion)
            # 3. ATR-based stop loss
            if (current_close > ema_trend or
                current_close >= dc_up or
                current_close > entry_price + 2.0 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals