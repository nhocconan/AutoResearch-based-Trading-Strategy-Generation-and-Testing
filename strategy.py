#!/usr/bin/env python3
"""
Experiment #020: 4h Donchian(20) Breakout + 1d Volume Spike + ATR Regime Filter

HYPOTHESIS: Donchian channel breakouts on 4h timeframe capture strong directional moves, 
confirmed by 1d volume spikes indicating institutional participation. The ATR-based regime 
filter (ATR(7)/ATR(30) > 1.2) ensures we only trade during elevated volatility periods, 
avoiding choppy markets. This combination has proven effective on SOLUSDT (test Sharpe 1.10-1.38) 
and adapts to both bull and bear markets by trading breakouts in the direction of elevated 
volatility. Targets 25-50 trades/year on 4h timeframe (100-200 total over 4 years) to minimize 
fee drag while capturing high-probability breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_vol_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
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
    
    # === 4h Indicators ===
    # Donchian Channel (20-period)
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:  # Need 20 periods for Donchian(20)
            donchian_h[i] = np.max(high[i-19:i+1])
            donchian_l[i] = np.min(low[i-19:i+1])
        else:
            donchian_h[i] = np.nan
            donchian_l[i] = np.nan
    
    # ATR-based volatility regime filter: ATR(7)/ATR(30) > 1.2 = elevated vol
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_7 = pd.Series(tr).ewm(span=7, min_periods=7, adjust=False).mean().values
    atr_30 = pd.Series(tr).ewm(span=30, min_periods=30, adjust=False).mean().values
    vol_regime = atr_7 / atr_30
    vol_regime[np.isnan(vol_regime) | (atr_30 == 0)] = 0.0
    
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
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(vol_regime[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade during elevated volatility (ATR(7)/ATR(30) > 1.2) ---
        high_volatility = vol_regime[i] > 1.2
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) on 1d ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate current ATR(14) for dynamic stoploss
            tr_segment = tr[max(0, i-13):i+1]  # Last 14 periods
            if len(tr_segment) >= 14:
                atr_14 = np.mean(tr_segment)
            else:
                atr_14 = atr_7[i]  # Fallback to ATR(7)
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price re-enters Donchian channel (failed breakout)
                if donchian_l[i] <= close[i] <= donchian_h[i]:
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
                # Exit if price re-enters Donchian channel (failed breakdown)
                if donchian_l[i] <= close[i] <= donchian_h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian H with volume and volatility confirmation
        long_condition = (
            close[i] > donchian_h[i] and 
            volume_spike and 
            high_volatility
        )
        
        # Short: Price breaks below Donchian L with volume and volatility confirmation
        short_condition = (
            close[i] < donchian_l[i] and 
            volume_spike and 
            high_volatility
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