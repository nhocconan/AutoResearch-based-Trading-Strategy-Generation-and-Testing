#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h EMA trend filter and volume confirmation.
# Bollinger Band squeeze (BB width < 20th percentile) indicates low volatility, often preceding breakouts.
# Direction determined by 12h EMA: long when price > EMA, short when price < EMA.
# Volume confirmation requires current volume > 2x 20-period average to avoid false breakouts.
# Designed to capture volatility breakouts in both bull and bear markets by aligning with trend.
# Targets 15-25 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h data
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Bollinger Bands (20, 2) on 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Middle band: 20-period SMA
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    # Bollinger Band width
    bb_width = (upper_bb - lower_bb) / sma_20
    # 20th percentile of BB width for squeeze detection (using expanding window to avoid look-ahead)
    bb_width_percentile = pd.Series(bb_width).expanding(min_periods=20).quantile(0.20).values
    # Squeeze condition: BB width < 20th percentile
    squeeze = bb_width < bb_width_percentile
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or 
            np.isnan(bb_width[i]) or 
            np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        squeeze_active = squeeze[i]
        ema_val = ema_12h_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Entry conditions: Bollinger Band squeeze + trend alignment + volume spike
            if squeeze_active:
                if price > ema_val and vol_spike:  # Long: price above 12h EMA + volume spike
                    signals[i] = 0.25
                    position = 1
                elif price < ema_val and vol_spike:  # Short: price below 12h EMA + volume spike
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: squeeze ends or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when squeeze ends or price breaks below 12h EMA
                if not squeeze_active or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when squeeze ends or price breaks above 12h EMA
                if not squeeze_active or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_BBSqueeze_12hEMA_Trend_Volume"
timeframe = "6h"
leverage = 1.0