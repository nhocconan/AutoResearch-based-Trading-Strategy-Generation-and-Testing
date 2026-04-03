#!/usr/bin/env python3
"""
Experiment #439: 6h ATR Channel Breakout with 12h Volume Spike and 1d Regime Filter

HYPOTHESAT: 6h ATR channel breakouts (price > upper/lower channel = close ± 1.5*ATR(20)) 
combined with 12h volume spike (>2x average) and 1d regime filter (price > EMA200 for long, 
< EMA200 for short) captures institutional breakouts with follow-through. The ATR channel 
adapts to volatility, volume confirms participation, and the 1d EMA200 filter ensures 
alignment with major trend. This avoids whipsaws in ranging markets while catching strong 
directional moves. Targets 15-25 trades/year on 6h timeframe (60-100 total over 4 years) 
to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_atr_breakout_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(200) on 1d close
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    else:
        ema_200_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Calculate ATR(20) for channel
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_20 = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # ATR Channel: upper/lower bands = close ± 1.5*ATR(20)
    upper_band = close + 1.5 * atr_20
    lower_band = close - 1.5 * atr_20
    
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
        if (np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in direction of 1d EMA200 ---
        price_above_1d_ema = close[i] > ema_200_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_200_1d_aligned[i]
        
        # --- Volume Confirmation: Require extreme volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss and channel re-entry) ---
        if in_position:
            # Calculate ATR(14) for stoploss (more responsive)
            tr_slice = np.zeros(i+1)
            tr_slice[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr_slice[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr_slice).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price re-enters the ATR channel (failed breakout)
                if close[i] < upper_band[i]:
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
                # Exit if price re-enters the ATR channel (failed breakdown)
                if close[i] > lower_band[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above upper ATR channel with volume spike in uptrend regime
        long_condition = (
            close[i] > upper_band[i] and 
            volume_spike and 
            price_above_1d_ema
        )
        
        # Short: Price breaks below lower ATR channel with volume spike in downtrend regime
        short_condition = (
            close[i] < lower_band[i] and 
            volume_spike and 
            price_below_1d_ema
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