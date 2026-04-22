#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian breakout with 1d RSI divergence filter and volume confirmation.
Long when price breaks above Donchian(20) high with bullish RSI divergence and volume spike.
Short when price breaks below Donchian(20) low with bearish RSI divergence and volume spike.
Exit when price returns to Donchian midpoint or RSI shows opposing divergence.
Uses 1d RSI for divergence detection to avoid false breakouts in ranging markets.
Designed for low trade frequency (15-30/year) to minimize fee drift.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for RSI divergence - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI on daily close
    close_d = pd.Series(df_daily['close'].values)
    delta = close_d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_d = 100 - (100 / (1 + rs))
    rsi_d = rsi_d.values
    
    # Align RSI to 6h timeframe
    rsi_d_aligned = align_htf_to_ltf(prices, df_daily, rsi_d)
    
    # Calculate 6h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after lookback periods
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(rsi_d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with bullish RSI divergence and volume spike
            bullish_div = (close[i] > donch_high[i] and 
                          rsi_d_aligned[i] > rsi_d_aligned[i-1] and  # RSI rising
                          close[i] > close[i-1] and  # Price rising
                          rsi_d_aligned[i] < 50)  # Not overbought
            volume_spike = volume[i] > 2.0 * vol_avg_20[i]
            
            if bullish_div and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with bearish RSI divergence and volume spike
            bearish_div = (close[i] < donch_low[i] and 
                          rsi_d_aligned[i] < rsi_d_aligned[i-1] and  # RSI falling
                          close[i] < close[i-1] and  # Price falling
                          rsi_d_aligned[i] > 50)  # Not oversold
            
            if bearish_div and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to midpoint OR RSI shows bearish divergence
                if close[i] <= donch_mid[i] or (rsi_d_aligned[i] < rsi_d_aligned[i-1] and close[i] < close[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to midpoint OR RSI shows bullish divergence
                if close[i] >= donch_mid[i] or (rsi_d_aligned[i] > rsi_d_aligned[i-1] and close[i] > close[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_RSI_Divergence_Volume"
timeframe = "6h"
leverage = 1.0
#%%