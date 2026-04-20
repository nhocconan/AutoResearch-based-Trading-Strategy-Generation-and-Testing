#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_Breakout_VolumeTrend_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: 20-period EMA for trend direction ===
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # === 4h: Calculate Donchian channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate rolling max/min with min_periods
    max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h: ATR(20) for volatility and stop loss ===
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # === 4h: Volume condition: current volume > 1.5x 20-period average volume ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get values
        ema_trend = ema_20_1d_aligned[i]
        upper = max_20[i]
        lower = min_20[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_vol = prices['volume'].iloc[i]
        current_vol_ma = vol_ma[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_trend) or np.isnan(upper) or np.isnan(lower) or 
            np.isnan(current_atr) or np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_condition = current_vol > 1.5 * current_vol_ma
        
        if position == 0:
            # Long conditions:
            # 1. Price above 1d EMA20 (uptrend)
            # 2. Price breaks above upper Donchian band with volume
            if (current_close > ema_trend and
                current_close > upper and
                vol_condition):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price below 1d EMA20 (downtrend)
            # 2. Price breaks below lower Donchian band with volume
            elif (current_close < ema_trend and
                  current_close < lower and
                  vol_condition):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below 1d EMA20 (trend change)
            # 2. Price hits lower Donchian band (mean reversion)
            # 3. ATR-based stop loss
            if (current_close < ema_trend or
                current_close <= lower or
                current_close < entry_price - 2.0 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above 1d EMA20 (trend change)
            # 2. Price hits upper Donchian band (mean reversion)
            # 3. ATR-based stop loss
            if (current_close > ema_trend or
                current_close >= upper or
                current_close > entry_price + 2.0 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals