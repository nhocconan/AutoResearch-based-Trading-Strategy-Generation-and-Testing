#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 and Bear Power increasing (less negative) in bullish 1w trend (close > EMA34_1w)
# Short when Bear Power < 0 and Bull Power decreasing (less positive) in bearish 1w trend (close < EMA34_1w)
# Uses discrete position sizing 0.25 to target ~12-37 trades/year and minimize fee drag
# Works in bull/bear markets: follows 1w trend with Elder Ray momentum confirmation

name = "6h_1w_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate 13-period EMA for Elder Ray (1d)
    def calculate_ema(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 2.0 / (period + 1)
        result = np.full(len(values), np.nan)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    ema13_1d = calculate_ema(close_1d, 13)
    
    # Calculate Elder Ray components (1d)
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema13_1d   # Bear Power = Low - EMA13
    
    # Calculate 34-period EMA for 1w trend filter
    ema34_1w = calculate_ema(close_1w, 34)
    
    # Calculate rate of change for Elder Ray to detect momentum shifts
    def calculate_roc(values, period=1):
        if len(values) < period + 1:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        result[period:] = (values[period:] - values[:-period]) / values[:-period]
        return result
    
    bull_power_roc_1d = calculate_roc(bull_power_1d, 1)
    bear_power_roc_1d = calculate_roc(bear_power_1d, 1)
    
    # Calculate 20-period average volume for confirmation (1d)
    if 'volume' in df_1d.columns:
        volume_1d = df_1d['volume'].values
    else:
        volume_1d = np.ones_like(close_1d)  # fallback
    
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    bull_power_roc_aligned = align_htf_to_ltf(prices, df_1d, bull_power_roc_1d)
    bear_power_roc_aligned = align_htf_to_ltf(prices, df_1d, bear_power_roc_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(bull_power_roc_aligned[i]) or np.isnan(bear_power_roc_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 1d volume (scaled)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # 1w trend filter: bullish if close > EMA34_1w, bearish if close < EMA34_1w
        bullish_trend = close[i] > ema34_1w_aligned[i]
        bearish_trend = close[i] < ema34_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if Elder Ray momentum deteriorates or trend changes
            if bull_power_roc_aligned[i] <= 0 or not bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Elder Ray momentum deteriorates or trend changes
            if bear_power_roc_aligned[i] >= 0 or not bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Bull Power positive AND increasing AND bullish 1w trend
            if (bull_power_aligned[i] > 0 and 
                bull_power_roc_aligned[i] > 0 and 
                bullish_trend and 
                volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Enter short: Bear Power negative AND decreasing AND bearish 1w trend
            elif (bear_power_aligned[i] < 0 and 
                  bear_power_roc_aligned[i] < 0 and 
                  bearish_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals