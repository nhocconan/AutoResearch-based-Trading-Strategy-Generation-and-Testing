#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h volume-weighted RSI with 1d trend filter
# Uses volume-weighted RSI to identify overextended moves with volume confirmation
# Trades mean reversion when VW-RSI < 30 (oversold) or > 70 (overbought) in direction of 1d trend
# Volume-weighted RSI gives more weight to price moves on high volume, filtering weak moves
# Target: 20-40 trades/year to avoid fee drag
name = "4h_VolWeightedRSI_1dTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA100 for trend filter (strong trend filter)
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Volume-weighted RSI (14-period)
    # Calculate price changes
    delta = np.diff(close, prepend=close[0])
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gains and losses
    vol_gains = gains * volume
    vol_losses = losses * volume
    
    # Calculate average volume-weighted gains and losses
    avg_vol_gain = pd.Series(vol_gains).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_vol_loss = pd.Series(vol_losses).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate VW-RSI
    rs = avg_vol_gain / (avg_vol_loss + 1e-10)
    vw_rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for VW-RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema100_1d_aligned[i]) or np.isnan(vw_rsi[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume filter: current volume > 1.3x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.3 * avg_volume
        
        if position == 0:
            # Long: VW-RSI oversold (<30) + volume + 1d uptrend
            if vw_rsi[i] < 30 and volume_filter and price > ema100_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: VW-RSI overbought (>70) + volume + 1d downtrend
            elif vw_rsi[i] > 70 and volume_filter and price < ema100_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: VW-RSI overbought (>70) or trend change
            if vw_rsi[i] > 70 or price < ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: VW-RSI oversold (<30) or trend change
            if vw_rsi[i] < 30 or price > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals