#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 1d trend filter
# - Long when price breaks above Donchian upper band + 12h volume > 1.5x average + price > 1d EMA50
# - Short when price breaks below Donchian lower band + 12h volume > 1.5x average + price < 1d EMA50
# - Exit when price returns to Donchian middle (20-period average) or trend reverses
# - Uses price channel breakouts which work in both trending and ranging markets
# - Volume confirmation ensures breakouts have conviction
# - Trend filter prevents counter-trend trades
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "4h_DonchianBreakout_12hVolume_1dTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    
    # 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or np.isnan(donchian_mid[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 20-period average (scaled to 4h)
        # 12h has 3x 4h bars, so multiply 4h volume by 3 to get equivalent 12h volume
        volume_12h_equiv = volume[i] * 3.0
        volume_filter = vol_ma_12h_aligned[i] > 0 and volume_12h_equiv > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Look for long entry: breakout above upper band + volume + uptrend
            if close[i] > highest_high[i] and volume_filter and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Look for short entry: breakdown below lower band + volume + downtrend
            elif close[i] < lowest_low[i] and volume_filter and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to middle or trend reverses
            if close[i] < donchian_mid[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to middle or trend reverses
            if close[i] > donchian_mid[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals