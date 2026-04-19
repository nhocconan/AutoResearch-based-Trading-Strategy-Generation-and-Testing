#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion using 4h Bollinger Bands with 1d trend filter and volume confirmation.
# Long when price touches lower BB (20,2) and closes above it, price > 1d EMA50, volume > 1.5x 4h avg volume.
# Short when price touches upper BB (20,2) and closes below it, price < 1d EMA50, volume > 1.5x 4h avg volume.
# Exit when price crosses 4h SMA20 (mean reversion complete) or reverses to opposite band.
# Uses Bollinger Bands for mean reversion zones, EMA for trend filter, volume for confirmation.
# Target: 15-30 trades/year per symbol to stay within frequency limits.
name = "1h_Bollinger_MeanReversion_EMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate Bollinger Bands on 4h close
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align Bollinger Bands to 1h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_4h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_4h, lower_bb)
    sma_20_aligned = align_htf_to_ltf(prices, df_4h, sma_20)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4h average volume for confirmation
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_bb = upper_bb_aligned[i]
        lower_bb = lower_bb_aligned[i]
        sma_20 = sma_20_aligned[i]
        ema_50 = ema_50_aligned[i]
        vol_ma = vol_ma_aligned[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price touches lower BB, closes above it, above EMA50, with volume confirmation
            if i > 0 and low[i-1] <= lower_bb_aligned[i-1] and close[i-1] < lower_bb_aligned[i-1] and \
               price > lower_bb and price > ema_50 and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Short entry: price touches upper BB, closes below it, below EMA50, with volume confirmation
            elif i > 0 and high[i-1] >= upper_bb_aligned[i-1] and close[i-1] > upper_bb_aligned[i-1] and \
                 price < upper_bb and price < ema_50 and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below SMA20 (mean reversion) or touches upper BB (reversal)
            if price < sma_20 or price >= upper_bb:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above SMA20 (mean reversion) or touches lower BB (reversal)
            if price > sma_20 or price <= lower_bb:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals