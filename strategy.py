#!/usr/bin/env python3
"""
Experiment #007: 6h Williams %R + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe. 
Entries occur when %R crosses above -20 (short) or below -80 (long) with confirmation 
from 1d volume spike (>2x average) and alignment with 1week EMA50 trend. 
This mean-reversion strategy with trend filter works in both bull and bear markets 
by capturing overextended moves that revert to the mean while avoiding counter-trend 
trades during strong trends. Targets 12-37 trades/year on 6h timeframe (50-150 total 
over 4 years) to minimize fee drag while exploiting short-term exhaustion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_007_6h_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Volume MA(20) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    else:
        vol_ma_20_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    williams_r = np.full(n, np.nan)
    
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i+1])
        lowest_low[i] = np.min(low[i-14:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # Neutral when range is zero
    
    # === 6h Indicators: ATR(14) for stoploss ===
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for HTF EMA50 and Williams %R
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require 1d volume spike (> 2.0x average) ---
        volume_spike = volume[i] > (2.0 * vol_ma_20_1d_aligned[i])
        
        # --- Trend Filter: Only trade counter to 1w EMA50 (mean reversion) ---
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # --- Williams %R Conditions ---
        wr_oversold = williams_r[i] < -80
        wr_overbought = williams_r[i] > -20
        wr_cross_up_oversold = (i > warmup and williams_r[i-1] >= -80 and wr_oversold)
        wr_cross_down_overbought = (i > warmup and williams_r[i-1] <= -20 and wr_overbought)
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit when Williams %R returns to neutral zone (take profit)
                if williams_r[i] > -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit when Williams %R returns to neutral zone (take profit)
                if williams_r[i] < -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Williams %R crosses below -80 + volume spike + price below 1w EMA50 (mean reversion in downtrend)
        long_condition = wr_cross_up_oversold and volume_spike and price_below_1w_ema
        
        # Short: Williams %R crosses above -20 + volume spike + price above 1w EMA50 (mean reversion in uptrend)
        short_condition = wr_cross_down_overbought and volume_spike and price_above_1w_ema
        
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