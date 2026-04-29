#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Donchian channels capture major breakouts aligned with weekly trend; volume >1.5x confirms participation
# Discrete sizing (0.25) minimizes fee churn; target 30-100 total trades over 4 years (7-25/year)

name = "1d_Donchian20_Breakout_1wEMA34_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.5 * vol_ma_30)
    
    # Precompute daily data for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30, 14, 34)  # warmup: need 20 bars for Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_30[i]) or
            np.isnan(daily_high_aligned[i]) or
            np.isnan(daily_low_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        
        # Use previous day's levels for breakout (shift by 1)
        if i >= 21:
            prev_high = daily_high_aligned[i-1]
            prev_low = daily_low_aligned[i-1]
            
            if not (np.isnan(prev_high) or np.isnan(prev_low)):
                # Calculate Donchian levels (20-period)
                upper_channel = np.max(daily_high_aligned[i-20:i])  # last 20 days excluding today
                lower_channel = np.min(daily_low_aligned[i-20:i])   # last 20 days excluding today
                
                if position == 0:  # Flat - look for new entries
                    if curr_volume_confirm:
                        # Bullish entry: price breaks above upper channel + above 1w EMA34
                        if curr_high > upper_channel and curr_close > curr_ema_34_1w:
                            signals[i] = 0.25
                            position = 1
                        # Bearish entry: price breaks below lower channel + below 1w EMA34
                        elif curr_low < lower_channel and curr_close < curr_ema_34_1w:
                            signals[i] = -0.25
                            position = -1
                
                elif position == 1:  # Long position
                    # Exit: price breaks below lower channel
                    if curr_low < lower_channel:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                
                elif position == -1:  # Short position
                    # Exit: price breaks above upper channel
                    if curr_high > upper_channel:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals