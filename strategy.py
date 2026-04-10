#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 12h trend filter, volume confirmation, and ATR trailing stop
# - Long when Williams %R(14) crosses above -80 (oversold) in 12h uptrend (close > EMA50) with volume > 1.3x 20-bar avg
# - Short when Williams %R(14) crosses below -20 (overbought) in 12h downtrend (close < EMA50) with volume spike
# - Uses ATR-based trailing stop: exit long when price drops 2.5*ATR from highest high since entry
# - Exit short when price rises 2.5*ATR from lowest low since entry
# - Discrete position sizing (±0.25) to minimize fee churn
# - Targets ~30 trades/year (120 total over 4 years) to avoid fee drag
# - Williams %R provides timely mean reversion signals in ranging markets, filtered by 12h trend

name = "4h_12h_williamsr_meanrev_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h Williams %R(14)
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r_12h = -100 * (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14 + 1e-10)
    williams_r_12h_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    
    # 12h volume confirmation: > 1.3x 20-period average
    avg_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.3 * avg_volume_20_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Pre-compute ATR for trailing stop (using 4h data)
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14 = pd.Series(true_ranges).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    atr_stop_multiplier = 2.5
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r_12h_aligned[i]) or 
            np.isnan(vol_spike_12h_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
            # ATR-based trailing stop: exit if price drops 2.5*ATR from highest high
            if prices['close'].iloc[i] < highest_since_entry - atr_stop_multiplier * atr_14[i]:
                position = 0
                highest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
            # ATR-based trailing stop: exit if price rises 2.5*ATR from lowest low
            if prices['close'].iloc[i] > lowest_since_entry + atr_stop_multiplier * atr_14[i]:
                position = 0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long signal: Williams %R crosses above -80 (from below) in 12h uptrend with volume spike
            if (i > 0 and 
                williams_r_12h_aligned[i-1] <= -80 and williams_r_12h_aligned[i] > -80 and
                prices['close'].iloc[i] > ema_50_12h_aligned[i] and 
                vol_spike_12h_aligned[i]):
                position = 1
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short signal: Williams %R crosses below -20 (from above) in 12h downtrend with volume spike
            elif (i > 0 and 
                  williams_r_12h_aligned[i-1] >= -20 and williams_r_12h_aligned[i] < -20 and
                  prices['close'].iloc[i] < ema_50_12h_aligned[i] and 
                  vol_spike_12h_aligned[i]):
                position = -1
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
    
    return signals