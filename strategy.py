#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extreme levels combined with 6h volume confirmation and 1d EMA50 trend filter.
# Long when 1d Williams %R < -80 (oversold) AND price breaks above 6h VWAP with volume spike (>1.8x 20-period volume MA) in 1d uptrend.
# Short when 1d Williams %R > -20 (overbought) AND price breaks below 6h VWAP with volume spike in 1d downtrend.
# Williams %R identifies exhaustion points; VWAP break confirms institutional participation; volume spike validates move.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years with discrete position sizing.

name = "6h_WilliamsR_Extreme_VWAP_Breakout_1dEMA50_VolumeSpike"
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
    open_prices = prices['open'].values
    
    # Get 1d data for Williams %R and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Williams %R extreme levels
    williams_r_oversold = williams_r < -80
    williams_r_overbought = williams_r > -20
    
    # Align Williams %R signals to lower timeframe (1d -> 6h)
    williams_r_oversold_aligned = align_htf_to_ltf(prices, df_1d, williams_r_oversold.astype(float))
    williams_r_overbought_aligned = align_htf_to_ltf(prices, df_1d, williams_r_overbought.astype(float))
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3
    tp_volume = typical_price * volume
    vwap = np.cumsum(tp_volume) / np.cumsum(volume)
    vwap = np.where(np.cumsum(volume) == 0, typical_price, vwap)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)  # Volume at least 1.8x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_oversold_aligned[i]) or np.isnan(williams_r_overbought_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vwap[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        open_val = open_prices[i]
        vol_spike = volume_spike[i]
        vwap_val = vwap[i]
        oversold_signal = bool(williams_r_oversold_aligned[i])
        overbought_signal = bool(williams_r_overbought_aligned[i])
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Williams %R oversold AND price breaks above VWAP AND volume spike AND 1d uptrend
            if oversold_signal and close_val > vwap_val and open_val <= vwap_val and vol_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND price breaks below VWAP AND volume spike AND 1d downtrend
            elif overbought_signal and close_val < vwap_val and open_val >= vwap_val and vol_spike and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: price closes below VWAP (mean reversion)
            if close_val < vwap_val:
                exit_signal = True
            # Exit: Williams %R becomes overbought (exhaustion)
            elif overbought_signal:
                exit_signal = True
            # Exit: 1d trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: price closes above VWAP (mean reversion)
            if close_val > vwap_val:
                exit_signal = True
            # Exit: Williams %R becomes oversold (exhaustion)
            elif oversold_signal:
                exit_signal = True
            # Exit: 1d trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals