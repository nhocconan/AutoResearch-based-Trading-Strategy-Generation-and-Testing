#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume spike >2.0x
# Donchian breakout catches momentum moves in both bull and bear markets
# 1w EMA50 ensures alignment with primary weekly trend; volume spike confirms institutional participation
# Discrete sizing (0.25) minimizes fee churn; target 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear: breakouts catch momentum moves, volume filter ensures legitimacy, EMA50 trend filter avoids counter-trend trades

name = "12h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    # Precompute weekly data for Donchian levels
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(240, 30, 14, 50)  # warmup: need 240 12h bars for weekly levels (20 weeks)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_30[i]) or
            np.isnan(weekly_high_aligned[i]) or
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(weekly_close_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        # Use previous week's levels (shift by 1)
        prev_high = weekly_high_aligned[i-1]
        prev_low = weekly_low_aligned[i-1]
        
        if position == 0:  # Flat - look for new entries
            if not (np.isnan(prev_high) or np.isnan(prev_low)):
                # Calculate Donchian levels (20-period)
                upper_channel = prev_high
                lower_channel = prev_low
                
                # Only trade with volume confirmation and trend filter
                if curr_volume_confirm:
                    # Bullish entry: price breaks above upper channel + above 1w EMA50
                    if curr_high > upper_channel and curr_close > curr_ema_50_1w:
                        signals[i] = 0.25
                        position = 1
                    # Bearish entry: price breaks below lower channel + below 1w EMA50
                    elif curr_low < lower_channel and curr_close < curr_ema_50_1w:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower channel
            if not (np.isnan(prev_high) or np.isnan(prev_low)):
                lower_channel = prev_low
                if curr_low < lower_channel:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper channel
            if not (np.isnan(prev_high) or np.isnan(prev_low)):
                upper_channel = prev_high
                if curr_high > upper_channel:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals