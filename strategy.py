#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume spike confirmation
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend via smoothed median prices.
# In trending markets, lines are separated and ordered (JAW > TEETH > LIPS for uptrend).
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades.
# Volume confirmation filters out false signals.
# Designed for fewer, higher-quality trades on 1d timeframe to minimize fee drag.
# Target: 15-25 trades/year for sustainable performance on BTC/ETH.

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
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
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 1d timeframe
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # JAW: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # TEETH: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # LIPS: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need sufficient history for 1w EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_spike = volume[i] > (volume_ma_20[i] * 2.0) if not np.isnan(volume_ma_20[i]) else False
        
        # Williams Alligator signals
        # Uptrend: Lips > Teeth > Jaw (green alignment)
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Downtrend: Lips < Teeth < Jaw (red alignment)
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: alligator uptrend, volume spike, price above 1w EMA50
            if alligator_long and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: alligator downtrend, volume spike, price below 1w EMA50
            elif alligator_short and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on alligator trend change or price below 1w EMA50
            if not alligator_long or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on alligator trend change or price above 1w EMA50
            if not alligator_short or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals