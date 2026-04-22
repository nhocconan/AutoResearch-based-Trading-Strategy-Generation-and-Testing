#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA34 trend filter
    # Donchian channels provide clear breakout levels. Volume confirms institutional interest.
    # 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades.
    # This combination reduces false breakouts and works in both bull and bear markets.
    # Target: 20-50 trades/year to minimize fee drag.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donch_period = 20
    upper_donch = pd.Series(high_4h).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_donch = pd.Series(low_4h).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Align Donchian channels to 4h timeframe (no alignment needed as we're already on 4h)
    upper_donch_4h = upper_donch
    lower_donch_4h = lower_donch
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter (20-period on 4h)
    vol_ma20_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma20_4h)
    vol_spike = df_4h['volume'].values > 1.5 * vol_ma20_4h  # Require 1.5x volume
    vol_spike_aligned = align_htf_to_ltf(prices, df_4h, vol_spike)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(upper_donch_4h[i]) or np.isnan(lower_donch_4h[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20_4h_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above upper Donchian with volume + price above 1d EMA34 (uptrend)
            if high[i] > upper_donch_4h[i] and vol_spike_aligned[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower Donchian with volume + price below 1d EMA34 (downtrend)
            elif low[i] < lower_donch_4h[i] and vol_spike_aligned[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian band or trend reversal vs 1d EMA34
            if position == 1:
                if low[i] < lower_donch_4h[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if high[i] > upper_donch_4h[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dEMA34_Volume_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0