#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeConfirmation
Hypothesis: On 6h timeframe, Camarilla R4/S4 breakouts with 1d EMA50 trend filter and volume confirmation capture strong momentum moves. R4/S4 levels represent strong breakout points where price often accelerates after breaking key resistance/support. In bull markets, we buy R4 breaks with uptrend; in bear markets, we sell S4 breaks with downtrend. Volume confirmation ensures breakout validity. Targets 12-30 trades/year to minimize fee drag while maintaining edge via trend filter and volume structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels on 1d (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla levels: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    # Using previous day's data to avoid look-ahead
    camarilla_r4 = close_1d_prev + ((high_1d_prev - low_1d_prev) * 1.1 / 2)
    camarilla_s4 = close_1d_prev - ((high_1d_prev - low_1d_prev) * 1.1 / 2)
    
    # Shift to get previous day's levels (avoid look-ahead)
    camarilla_r4 = np.roll(camarilla_r4, 1)
    camarilla_s4 = np.roll(camarilla_s4, 1)
    camarilla_r4[0] = np.nan
    camarilla_s4[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume average (20-period = ~5 days on 6h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(20, 50, 1)  # volume MA, 1d EMA, Camarilla shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_1d_val = ema_50_1d_aligned[i]
        camarilla_r4_val = camarilla_r4_aligned[i]
        camarilla_s4_val = camarilla_s4_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = vol_val > 1.3 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above camarilla R4 with uptrend (close > EMA50) and volume confirmation
            long_signal = (high_val > camarilla_r4_val) and (close_val > ema_50_1d_val) and volume_confirmed
            # Short: price breaks below camarilla S4 with downtrend (close < EMA50) and volume confirmation
            short_signal = (low_val < camarilla_s4_val) and (close_val < ema_50_1d_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below camarilla S4 (exit long)
            if low_val < camarilla_s4_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses below EMA50
            elif close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above camarilla R4 (exit short)
            if high_val > camarilla_r4_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses above EMA50
            elif close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0