#!/usr/bin/env python3
"""
Experiment #019: 6h ATR Breakout + Volume Spike + 12h Trend Filter

HYPOTHESIS: Breakouts above/below ATR-based channels on 6h timeframe, confirmed by 
12h volume spikes (>2.0x average) and aligned with 12h trend (price > EMA50 for long, 
< EMA50 for short), captures high-probability momentum moves. ATR channels adapt to 
volatility, volume confirms institutional participation, and 12h trend filter avoids 
counter-trend trades. Targets 12-37 trades/year (50-150 total over 4 years) on 6h 
timeframe to minimize fee drag while capturing sustained moves in both bull and bear 
markets. Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_atr_breakout_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(50) on 12h close
    if len(df_12h) >= 50:
        close_12h = df_12h['close'].values
        ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    else:
        ema_50_12h_aligned = np.full(n, np.nan)
    
    # === HTF: 12h data for volume spike confirmation (Call ONCE before loop) ===
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === 6h Indicators ===
    # Calculate ATR(14) for channel width
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate ATR-based channels (similar to Keltner)
    ema_20_close = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    upper_channel = ema_20_close + 2.0 * atr_14
    lower_channel = ema_20_close - 2.0 * atr_14
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: 12h EMA50 direction ---
        price_above_12h_ema = close[i] > ema_50_12h_aligned[i]
        price_below_12h_ema = close[i] < ema_50_12h_aligned[i]
        
        # --- Volume Confirmation: Require extreme volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate current ATR for stoploss
            if i >= 14:
                # Recalculate ATR for current point using recent data
                tr_recent = np.zeros(i+1)
                tr_recent[0] = high[0] - low[0]
                for j in range(1, i+1):
                    tr_recent[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_current = pd.Series(tr_recent).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            else:
                atr_current = atr_14[i]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_current
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at upper channel (trailing)
                if close[i] >= upper_channel[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_current
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at lower channel (trailing)
                if close[i] <= lower_channel[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above upper channel with volume spike and 12h uptrend
        long_condition = (
            close[i] > upper_channel[i] and 
            volume_spike and 
            price_above_12h_ema
        )
        
        # Short: Break below lower channel with volume spike and 12h downtrend
        short_condition = (
            close[i] < lower_channel[i] and 
            volume_spike and 
            price_below_12h_ema
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