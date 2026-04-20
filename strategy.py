#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Volume_Weighted_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1w: 34-period EMA for long-term trend direction ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d: VWAP for intraday fair value ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Typical price
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    # VWAP = sum(tp * volume) / sum(volume)
    vwap_1d = np.cumsum(tp_1d * volume_1d) / np.cumsum(volume_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === 12h: ATR(14) for volatility and stop loss ===
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
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        ema_trend = ema_34_1w_aligned[i]
        vwap = vwap_1d_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_trend) or np.isnan(vwap) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === Volume condition: current volume > 1.5x 20-period 12h average volume ===
        if i >= 20:
            vol_ma = np.mean(prices['volume'].iloc[i-20:i].values)
            vol_condition = current_volume > 1.5 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long conditions:
            # 1. Price above 1w EMA34 (long-term uptrend)
            # 2. Price above 1d VWAP (intraday bullish)
            # 3. Volume confirmation
            if (current_close > ema_trend and
                current_close > vwap and
                vol_condition):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price below 1w EMA34 (long-term downtrend)
            # 2. Price below 1d VWAP (intraday bearish)
            # 3. Volume confirmation
            elif (current_close < ema_trend and
                  current_close < vwap and
                  vol_condition):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below 1w EMA34 (trend change)
            # 2. Price falls below 1d VWAP (intraday bearish)
            # 3. ATR-based stop loss
            if (current_close < ema_trend or
                current_close < vwap or
                current_close < entry_price - 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above 1w EMA34 (trend change)
            # 2. Price rises above 1d VWAP (intraday bullish)
            # 3. ATR-based stop loss
            if (current_close > ema_trend or
                current_close > vwap or
                current_close > entry_price + 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals