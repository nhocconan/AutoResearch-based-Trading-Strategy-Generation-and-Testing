#!/usr/bin/env python3
"""
4h_Volume_Weighted_CCI_Trend_Signal
Hypothesis: On 4h timeframe, use Commodity Channel Index (CCI) with volume-weighted adjustment to detect trend exhaustion and reversal points. Combine with 12h trend filter and volume confirmation to avoid whipsaws. Works in bull markets by buying pullbacks in uptrends and in bear markets by selling rallies in downtrends. Target 25-40 trades/year via strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_cci(high, low, close, period=20):
    """Calculate Commodity Channel Index"""
    typical_price = (high + low + close) / 3.0
    sma_tp = np.zeros_like(typical_price)
    mad = np.zeros_like(typical_price)
    
    if len(typical_price) < period:
        return np.full_like(typical_price, np.nan)
    
    for i in range(len(typical_price)):
        if i < period - 1:
            sma_tp[i] = np.nan
            mad[i] = np.nan
        else:
            sma_tp[i] = np.mean(typical_price[i-period+1:i+1])
            mad[i] = np.mean(np.abs(typical_price[i-period+1:i+1] - sma_tp[i]))
    
    cci = np.full_like(typical_price, np.nan)
    valid = (~np.isnan(sma_tp)) & (~np.isnan(mad)) & (mad != 0)
    cci[valid] = (typical_price[valid] - sma_tp[valid]) / (0.015 * mad[valid])
    return cci

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume-weighted CCI: typical price weighted by volume
    typical_price = (high_4h + low_4h + close_4h) / 3.0
    vol_tp = typical_price * volume_4h
    
    # Calculate VW-CCI (volume-weighted CCI)
    vw_cci = calculate_cci(high_4h, low_4h, close_4h, 20)  # Standard CCI calculation
    # Adjust for volume: when volume is high, CCI signal is stronger
    vol_factor = np.zeros_like(volume_4h)
    if len(volume_4h) >= 20:
        vol_ma = np.zeros_like(volume_4h)
        for i in range(len(volume_4h)):
            if i < 19:
                vol_ma[i] = np.nan
            else:
                vol_ma[i] = np.mean(volume_4h[i-19:i+1])
        vol_factor = np.where(volume_4h > vol_ma, 1.2, 0.8)  # Amplify in high volume
    else:
        vol_factor = np.ones_like(volume_4h)
    
    vw_cci_adjusted = vw_cci * vol_factor
    vw_cci_adjusted_aligned = align_htf_to_ltf(prices, df_4h, vw_cci_adjusted)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA34 for trend
    ema34_12h = np.zeros_like(close_12h)
    if len(close_12h) >= 34:
        ema34_12h[33] = np.mean(close_12h[:34])
        multiplier = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema34_12h[i] = (close_12h[i] - ema34_12h[i-1]) * multiplier + ema34_12h[i-1]
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: 4h volume > 1.5 * 20-period average
    vol_ma_4h = np.zeros_like(volume_4h)
    for i in range(len(volume_4h)):
        if i < 19:
            vol_ma_4h[i] = np.nan
        else:
            vol_ma_4h[i] = np.mean(volume_4h[i-19:i+1])
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    volume_confirmed = volume_4h > (1.5 * vol_ma_4h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if indicators not ready
        if (np.isnan(vw_cci_adjusted_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume_ok = volume_confirmed[i] if i < len(volume_confirmed) else False
        
        if position == 0:
            # Long: VW-CCI oversold (< -100) in uptrend (price > 12h EMA34) with volume
            if (vw_cci_adjusted_aligned[i] < -100 and 
                price > ema34_12h_aligned[i] and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short: VW-CCI overbought (> 100) in downtrend (price < 12h EMA34) with volume
            elif (vw_cci_adjusted_aligned[i] > 100 and 
                  price < ema34_12h_aligned[i] and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: VW-CCI returns to neutral (> -50) or trend breakdown
            if vw_cci_adjusted_aligned[i] > -50 or price < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: VW-CCI returns to neutral (< 50) or trend reversal
            if vw_cci_adjusted_aligned[i] < 50 or price > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Volume_Weighted_CCI_Trend_Signal"
timeframe = "4h"
leverage = 1.0