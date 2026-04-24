#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 80-120 total trades over 4 years (20-30/year).
- HTF: 1d for EMA50 trend filter and Williams %R calculation (provides longer-term context).
- Entry: Long when Williams %R(14) crosses above -80 (oversold) AND price > 1d EMA50 AND volume > 1.5 * 6h volume MA(20);
         Short when Williams %R(14) crosses below -20 (overbought) AND price < 1d EMA50 AND volume > 1.5 * 6h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via trend filter (signal=0 when price closes below/above 1d EMA50).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R captures short-term extremes; 1d EMA50 filters counter-trend trades; volume spike confirms momentum behind reversal.
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend) with volume confirmation to avoid false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for Williams %R(14) and EMA(50)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate EMA(50) on 1d
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 6h data for volume MA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    
    # Calculate volume MA(20) on 6h
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need enough for Williams %R(14) and EMA(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if price closes below/above 1d EMA50 (trend filter)
        if position == 1:
            if curr_close < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Williams %R conditions with volume confirmation
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1] if i > 0 else wr
        
        # Oversold condition: Williams %R crosses above -80
        oversold_cross = (wr_prev <= -80) and (wr > -80)
        # Overbought condition: Williams %R crosses below -20
        overbought_cross = (wr_prev >= -20) and (wr < -20)
        
        # Trend filter from 1d EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R crosses above -80 (oversold) AND price above 1d EMA50
                if oversold_cross and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 (overbought) AND price below 1d EMA50
                elif overbought_cross and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_MeanReversion_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0