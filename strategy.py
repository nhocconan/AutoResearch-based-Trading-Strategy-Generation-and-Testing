#!/usr/bin/env python3
"""
Experiment #279: 6h Williams %R + 12h EMA Trend + Volume Spike Confirmation

HYPOTHESIS: Williams %R identifies overextended conditions on 6h timeframe, while 12h EMA provides medium-term trend direction and volume spikes confirm institutional participation. This combination should work in both bull and bear markets by capturing mean-reversion moves aligned with the intermediate trend. Targets 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to minimize fee drag while focusing on high-probability reversals at trend extremes with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(30) on 12h close
    if len(df_12h) >= 30:
        close_12h = df_12h['close'].values
        ema_30_12h = pd.Series(close_12h).ewm(span=30, min_periods=30, adjust=False).mean().values
        ema_30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_30_12h)
    else:
        ema_30_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    valid = (highest_high_14 - lowest_low_14) != 0
    williams_r[valid] = -100 * (highest_high_14[valid] - close[valid]) / (highest_high_14[valid] - lowest_low_14[valid])
    
    # Volume Spike: Current volume > 2.0 * average volume over last 20 periods
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(ema_30_12h_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
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
                stop_level = close[entry_bar] - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Williams %R > -50 (mean reversion halfway)
                if williams_r[i] > -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = close[entry_bar] + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Williams %R < -50 (mean reversion halfway)
                if williams_r[i] < -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Price trend alignment with 12h EMA
        price_above_ema = close[i] > ema_30_12h_aligned[i]
        price_below_ema = close[i] < ema_30_12h_aligned[i]
        
        # Long: Oversold Williams %R with price above 12h EMA and volume spike
        if oversold and price_above_ema and volume_spike[i]:
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        # Short: Overbought Williams %R with price below 12h EMA and volume spike
        elif overbought and price_below_ema and volume_spike[i]:
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals