#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action reverses at 12h VWAP when aligned with 1d trend and volume spike.
# VWAP acts as dynamic support/resistance where institutional traders operate.
# In trending markets, price pulls back to VWAP before continuing trend.
# In ranging markets, VWAP acts as mean reversion level.
# Combined with 1d trend filter and volume confirmation for high-probability entries.
# Targets 20-35 trades/year with controlled risk.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for VWAP calculation (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h VWAP (typical price * volume / cumulative volume)
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    vwap_numerator = np.cumsum(typical_price_12h * volume_12h)
    vwap_denominator = np.cumsum(volume_12h)
    vwap_12h = vwap_numerator / vwap_denominator
    
    # Align 12h VWAP to 4h timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # 1d trend filter: EMA(50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vwap_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price pulls back to VWAP support in uptrend with volume
            if (close[i] > ema_50_1d_aligned[i] and  # Uptrend filter
                low[i] <= vwap_12h_aligned[i] * 1.005 and  # Touch or slightly below VWAP
                close[i] > vwap_12h_aligned[i] and  # Close above VWAP (confirmation)
                volume[i] > 2.5 * vol_avg_20[i]):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price rallies to VWAP resistance in downtrend with volume
            elif (close[i] < ema_50_1d_aligned[i] and  # Downtrend filter
                  high[i] >= vwap_12h_aligned[i] * 0.995 and  # Touch or slightly above VWAP
                  close[i] < vwap_12h_aligned[i] and  # Close below VWAP (confirmation)
                  volume[i] > 2.5 * vol_avg_20[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price breaks below VWAP or trend reverses
                if (close[i] < vwap_12h_aligned[i] * 0.995 or  # Clear break below VWAP
                    close[i] < ema_50_1d_aligned[i]):  # Trend break
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price breaks above VWAP or trend reverses
                if (close[i] > vwap_12h_aligned[i] * 1.005 or  # Clear break above VWAP
                    close[i] > ema_50_1d_aligned[i]):  # Trend break
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Pullback_12hVWAP_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0