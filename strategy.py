#!/usr/bin/env python3
"""
Experiment #049: 4h Donchian(20) Breakout + 1d Trend Filter + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 4h capture significant price moves with 
institutional participation. The 1d EMA50 filter ensures alignment with higher timeframe 
trend, reducing false breakouts. Volume confirmation (>1.5x 20-period average) adds 
conviction. Discrete position sizing (0.25) and ATR-based stoploss (2.5x) manage risk. 
Target: 100-180 trades over 4 years (25-45/year) to balance opportunity with fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_vol_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 4h Donchian Channel (20-period) ===
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= lookback:
        for i in range(lookback, n):
            donchian_high[i] = np.max(high[i-lookback:i])
            donchian_low[i] = np.min(low[i-lookback:i])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, lookback)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long when price breaks above Donchian high with volume and trend alignment
        long_condition = (
            close[i] > donchian_high[i] and 
            vol_ratio_1d_aligned[i] > 1.5 and 
            close[i] > ema_50_1d_aligned[i]
        )
        
        # Short when price breaks below Donchian low with volume and trend alignment
        short_condition = (
            close[i] < donchian_low[i] and 
            vol_ratio_1d_aligned[i] > 1.5 and 
            close[i] < ema_50_1d_aligned[i]
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>