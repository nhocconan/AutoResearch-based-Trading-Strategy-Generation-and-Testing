#!/usr/bin/env python3
"""
Experiment #4779: 6h Elder Ray + Regime Filter + Volume Spike
HYPOTHESIS: On 6h timeframe, Elder Ray (Bull/Bear Power) combined with ADX regime filter and volume confirmation captures strong momentum moves while avoiding whipsaws. Bull Power > 0 + Bear Power < 0 indicates bullish momentum; Bear Power > 0 + Bull Power < 0 indicates bearish momentum. ADX > 25 filters for trending markets. Volume > 1.5x average confirms conviction. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (strong uptrends) and bear markets (strong downtrends) by requiring clear directional momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4779_6h_elder_ray_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for trend context (optional filter)
    df_12h = get_htf_data(prices, '12h')
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray Components ===
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # === 6h Indicators: ADX(14) for regime filter ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))]
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_ma = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # ADX
    dx = np.where((di_plus + di_minus) > 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume confirmation ===
    vol_ma = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(13, 14, 20)  # EMA13, ADX, Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or loss of momentum ---
        if in_position:
            # Exit conditions: momentum reversal or ADX weakening
            if position_side > 0:  # Long
                # Exit if bear power becomes positive (momentum fading) OR ADX < 20 (trend weakening)
                if bear_power[i] > 0 or adx[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit if bull power becomes positive (momentum fading) OR ADX < 20 (trend weakening)
                if bull_power[i] > 0 or adx[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Elder Ray momentum conditions with ADX filter
        # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND ADX > 25 (trending)
        long_momentum = (bull_power[i] > 0) and (bear_power[i] < 0) and (adx[i] > 25)
        # Short: Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND ADX > 25 (trending)
        short_momentum = (bear_power[i] > 0) and (bull_power[i] < 0) and (adx[i] > 25)
        
        # Final entry conditions
        if long_momentum and vol_confirm:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_momentum and vol_confirm:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals