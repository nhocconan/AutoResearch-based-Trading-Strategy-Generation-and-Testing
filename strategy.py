#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_1dTrend_VolumeSpike_DynamicSize
Hypothesis: On 6h timeframe, breakouts at Camarilla R4/S4 levels (stronger levels) with 1d EMA50 trend filter and volume spike (>1.5x average) capture significant moves. Uses dynamic position sizing based on volatility (inverse ATR) to reduce risk during high volatility periods. Works in both bull and bear markets by following 1d trend direction, avoiding counter-trend trades. Targets 12-30 trades/year (50-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need warmup for EMA, volume, ATR
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter (more stable than EMA34)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R4 and S4 levels (breakout/continuation levels)
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (1 bar delay for completed 1d bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate average volume for confirmation (24-period SMA for 6x4h=1d equivalent)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate ATR (20) for volatility-based position sizing and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Start after warmup (need 50 for EMA, 24 for volume, 20 for ATR)
    start_idx = max(50, 24, 20)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1d_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r4_val) or 
            np.isnan(s4_val) or np.isnan(atr_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume (strong breakout)
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Dynamic position sizing: inverse volatility (smaller size when ATR high)
        # Normalize ATR to 6h median ATR for scaling
        if i >= 100:  # Need history for median
            historical_atr = atr[max(start_idx, i-100):i]
            median_atr = np.nanmedian(historical_atr)
            if median_atr > 0:
                vol_scaling = min(1.5, max(0.5, median_atr / atr_val))  # Inverse volatility scaling
            else:
                vol_scaling = 1.0
        else:
            vol_scaling = 1.0
        
        dynamic_size = base_size * vol_scaling
        # Cap size at 0.35 to respect limits
        dynamic_size = min(0.35, dynamic_size)
        
        # Long logic: price breaks above Camarilla R4 with 1d uptrend and volume confirmation
        long_condition = (close_val > r4_val) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below Camarilla S4 with 1d downtrend and volume confirmation
        short_condition = (close_val < s4_val) and (close_val < ema_val) and volume_confirmed
        
        # Stoploss logic: 3x ATR from entry (wider stop for 6h timeframe)
        stoploss_long = position == 1 and close_val < entry_price - 3.0 * atr_val
        stoploss_short = position == -1 and close_val > entry_price + 3.0 * atr_val
        
        if long_condition and position != 1:
            signals[i] = dynamic_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -dynamic_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif position == 1 and (stoploss_long or close_val < ema_val):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (stoploss_short or close_val > ema_val):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = dynamic_size
            else:
                signals[i] = -dynamic_size
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1dTrend_VolumeSpike_DynamicSize"
timeframe = "6h"
leverage = 1.0