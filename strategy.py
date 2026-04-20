#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_Confluence_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly: 20-period EMA for trend direction ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Daily: ATR(14) for volatility and stop loss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        ema_trend = ema_20_1w_aligned[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        current_atr = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(current_atr):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === Volume condition: current volume > 1.5x 10-day average volume ===
        if i >= 10:
            vol_ma = np.mean(prices['volume'].iloc[i-10:i].values)
            vol_condition = current_volume > 1.5 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long conditions:
            # 1. Price above weekly EMA20 (uptrend)
            # 2. Volume confirmation
            # 3. Price above daily open (bullish bias)
            if (current_close > ema_trend and
                vol_condition and
                current_close > prices['open'].iloc[i]):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price below weekly EMA20 (downtrend)
            # 2. Volume confirmation
            # 3. Price below daily open (bearish bias)
            elif (current_close < ema_trend and
                  vol_condition and
                  current_close < prices['open'].iloc[i]):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below weekly EMA20 (trend change)
            # 2. ATR-based stop loss
            if (current_close < ema_trend or
                current_close < entry_price - 2.0 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above weekly EMA20 (trend change)
            # 2. ATR-based stop loss
            if (current_close > ema_trend or
                current_close > entry_price + 2.0 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals