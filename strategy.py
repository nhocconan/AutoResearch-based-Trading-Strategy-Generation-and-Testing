#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d trend filter + volume confirmation
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# - Trend up: Lips > Teeth > Jaw (all aligned upward)
# - Trend down: Lips < Teeth < Jaw (all aligned downward)
# - Enter long when Alligator signals uptrend + price > Jaw + 1d EMA200 up + volume spike
# - Enter short when Alligator signals downtrend + price < Jaw + 1d EMA200 down + volume spike
# - Exit when Alligator reverses or price crosses Jaw in opposite direction
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-30 trades/year (50-120 total over 4 years) to avoid fee drag
# - Works in both bull and bear: Alligator adapts to trending markets, volume filter avoids false signals

name = "12h_1d_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1d EMA(200) slope for trend direction (up/down)
    ema_200_slope_1d = np.zeros_like(ema_200_1d)
    ema_200_slope_1d[1:] = ema_200_1d[1:] - ema_200_1d[:-1]
    ema_200_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_slope_1d)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Williams Alligator on 12h timeframe
    # Median price = (high + low) / 2
    median_price = (prices['high'] + prices['low']) / 2
    
    # Jaw: 13-period SMA, smoothed by 8 periods
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = pd.Series(jaw).rolling(window=8, min_periods=8).mean()
    jaw_values = jaw.values
    
    # Teeth: 8-period SMA, smoothed by 5 periods
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = pd.Series(teeth).rolling(window=5, min_periods=5).mean()
    teeth_values = teeth.values
    
    # Lips: 5-period SMA, smoothed by 3 periods
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = pd.Series(lips).rolling(window=3, min_periods=3).mean()
    lips_values = lips.values
    
    # Alligator alignment signals
    # Bullish: Lips > Teeth > Jaw (all aligned upward)
    # Bearish: Lips < Teeth < Jaw (all aligned downward)
    alligator_bull = (lips_values > teeth_values) & (teeth_values > jaw_values)
    alligator_bear = (lips_values < teeth_values) & (teeth_values < jaw_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(ema_200_slope_1d_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or np.isnan(lips_values[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator reverses bearish OR price crosses below Jaw (trend invalidation)
            if alligator_bear[i] or prices['close'].iloc[i] < jaw_values[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator reverses bullish OR price crosses above Jaw (trend invalidation)
            if alligator_bull[i] or prices['close'].iloc[i] > jaw_values[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with 1d trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Long signal: Alligator bullish + 1d EMA200 sloping up + price above Jaw
                if (alligator_bull[i] and 
                    ema_200_slope_1d_aligned[i] > 0 and 
                    prices['close'].iloc[i] > jaw_values[i]):
                    position = 1
                    signals[i] = 0.25
                # Short signal: Alligator bearish + 1d EMA200 sloping down + price below Jaw
                elif (alligator_bear[i] and 
                      ema_200_slope_1d_aligned[i] < 0 and 
                      prices['close'].iloc[i] < jaw_values[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals