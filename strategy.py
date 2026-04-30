#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian channels capture structural breaks; 1d EMA34 ensures alignment with daily trend
# Volume > 1.8x 20-period average confirms institutional participation
# Discrete sizing (0.25) minimizes fee churn; target 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear: breakouts catch momentum moves, volume filter ensures legitimacy, daily EMA trend filter avoids counter-trend trades

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Calculate 1d EMA34 for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # Precompute daily data for Donchian channels (20-period)
    df_1d_dc = get_htf_data(prices, '1d')
    if len(df_1d_dc) < 20:
        return np.zeros(n)
    
    daily_high = df_1d_dc['high'].values
    daily_low = df_1d_dc['low'].values
    
    # Calculate Donchian channels: 20-period high/low
    high_20 = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d_dc, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d_dc, low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 14, 34)  # warmup: need 100 12h bars for Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(high_20_aligned[i]) or
            np.isnan(low_20_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_high_20 = high_20_aligned[i]
        curr_low_20 = low_20_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if not (np.isnan(curr_high_20) or np.isnan(curr_low_20) or np.isnan(curr_ema_34_1d)):
                # Only trade with volume confirmation and trend filter
                if curr_volume_confirm:
                    # Bullish entry: price breaks above upper Donchian + above 1d EMA34
                    if curr_high > curr_high_20 and curr_close > curr_ema_34_1d:
                        signals[i] = 0.25
                        position = 1
                    # Bearish entry: price breaks below lower Donchian + below 1d EMA34
                    elif curr_low < curr_low_20 and curr_close < curr_ema_34_1d:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian
            if not np.isnan(curr_low_20):
                if curr_low < curr_low_20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian
            if not np.isnan(curr_high_20):
                if curr_high > curr_high_20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals