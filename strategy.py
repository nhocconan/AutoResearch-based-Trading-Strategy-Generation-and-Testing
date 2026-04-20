#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Adaptive_Breakout_With_Pullback"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === Daily: 20-period EMA for trend direction ===
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # === Weekly: 20-period EMA for higher timeframe trend filter ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === 4h: 20-period Donchian channels for breakout signals ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 4-period lookback Donchian channels
    # Using rolling window with shift(1) to avoid look-ahead
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 4h: ATR(14) for volatility and dynamic sizing ===
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h: 20-period EMA for pullback entry filter ===
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        ema_trend_1d = ema_20_1d_aligned[i]
        ema_trend_1w = ema_20_1w_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_trend_1d) or np.isnan(ema_trend_1w) or 
            np.isnan(donch_high) or np.isnan(donch_low) or 
            np.isnan(current_atr) or np.isnan(ema_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === Volume condition: current volume > 1.5x 20-period 4h average volume ===
        if i >= 20:
            vol_ma = np.mean(prices['volume'].iloc[i-20:i].values)
            vol_condition = current_volume > 1.5 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long conditions:
            # 1. Price above daily EMA20 (daily uptrend)
            # 2. Price above weekly EMA20 (weekly uptrend filter)
            # 3. Price breaks above 4h Donchian high with volume
            if (current_close > ema_trend_1d and
                current_close > ema_trend_1w and
                current_close > donch_high and
                vol_condition):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price below daily EMA20 (daily downtrend)
            # 2. Price below weekly EMA20 (weekly downtrend filter)
            # 3. Price breaks below 4h Donchian low with volume
            elif (current_close < ema_trend_1d and
                  current_close < ema_trend_1w and
                  current_close < donch_low and
                  vol_condition):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below daily EMA20 (trend change)
            # 2. Price hits 4h Donchian low (mean reversion target)
            # 3. ATR-based stop loss
            if (current_close < ema_trend_1d or
                current_close <= donch_low or
                current_close < entry_price - 2.0 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above daily EMA20 (trend change)
            # 2. Price hits 4h Donchian high (mean reversion target)
            # 3. ATR-based stop loss
            if (current_close > ema_trend_1d or
                current_close >= donch_high or
                current_close > entry_price + 2.0 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals