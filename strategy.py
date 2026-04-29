#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly EMA34 trend filter and volume confirmation
# Donchian channels capture volatility-based breakouts; weekly EMA34 ensures alignment with long-term trend
# Volume > 1.8x 24-period average confirms institutional participation
# Discrete sizing (0.25) minimizes fee churn; target 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear: breakouts catch momentum moves, trend filter avoids counter-trend trades, volume filter ensures legitimacy

name = "12h_Donchian20_Breakout_1wEMA34_Trend_VolumeConfirm_v1"
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
    
    # Calculate weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.8x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_ma_24)
    
    # Precompute daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)  # warmup: need 24 for volume MA, 34 for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_24[i]) or
            np.isnan(daily_high_aligned[i]) or
            np.isnan(daily_low_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        
        # Use previous day's levels (shift by 1)
        prev_high = daily_high_aligned[i-1]
        prev_low = daily_low_aligned[i-1]
        
        if position == 0:  # Flat - look for new entries
            if not (np.isnan(prev_high) or np.isnan(prev_low)):
                # Calculate Donchian levels (20-period)
                upper_channel = prev_high
                lower_channel = prev_low
                
                # Only trade with volume confirmation and trend filter
                if curr_volume_confirm:
                    # Bullish entry: price breaks above upper channel + above weekly EMA34
                    if curr_high > upper_channel and curr_close > curr_ema_34_1w:
                        signals[i] = 0.25
                        position = 1
                    # Bearish entry: price breaks below lower channel + below weekly EMA34
                    elif curr_low < lower_channel and curr_close < curr_ema_34_1w:
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