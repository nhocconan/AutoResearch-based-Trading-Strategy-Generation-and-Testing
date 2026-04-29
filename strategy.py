#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above 20-period Donchian high AND close > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below 20-period Donchian low AND close < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when price crosses back inside the Donchian channel (mean reversion of breakout failure)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 19-50 trades/year on 4h.
# Donchian channels provide objective breakout levels that work in trending markets.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# 1d EMA34 filter ensures we only trade with the higher timeframe trend.
# Works in bull markets via upward breakouts, works in bear via downward breakouts with volume spikes.

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter for appropriate trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_34_1d_aligned[i]
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        curr_close = close[i]
        prev_close = close[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND close > 1d EMA34 AND volume confirmation
            if curr_close > dc_high and prev_close <= dc_high and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND close < 1d EMA34 AND volume confirmation
            elif curr_close < dc_low and prev_close >= dc_low and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses back inside Donchian channel
            if curr_close < dc_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses back inside Donchian channel
            if curr_close > dc_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals