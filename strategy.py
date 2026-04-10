#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 1d Volume and Trend Filter
# - Primary: 6h timeframe for balance of signal frequency and fee control
# - HTF: 1d for trend (EMA50) and volume confirmation
# - Long: Williams %R(14) < -80 (oversold) + price > 1d EMA50 (uptrend) + volume > 1.5x 20-period MA
# - Short: Williams %R(14) > -20 (overbought) + price < 1d EMA50 (downtrend) + volume > 1.5x 20-period MA
# - Exit: Williams %R crosses above -50 (long exit) or below -50 (short exit)
# - Position sizing: 0.25 (discrete level)
# - Target: 80-180 total trades over 4 years (20-45/year) - within 6h sweet spot
# - Works in bull/bear: Mean reversion in ranging markets (2025), trend filter avoids counter-trend in strong moves

name = "6h_1d_williamsr_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Williams %R(14)
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) + uptrend (price > EMA50) + volume spike
            if (williams_r[i] < -80 and 
                close_6h[i] > ema_50_1d_aligned[i] and 
                volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought (> -20) + downtrend (price < EMA50) + volume spike
            elif (williams_r[i] > -20 and 
                  close_6h[i] < ema_50_1d_aligned[i] and 
                  volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R crosses above -50 (long exit) or below -50 (short exit)
            if position == 1:  # Long position
                if williams_r[i] > -50:  # Exit long when Williams %R crosses above -50
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r[i] < -50:  # Exit short when Williams %R crosses below -50
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals