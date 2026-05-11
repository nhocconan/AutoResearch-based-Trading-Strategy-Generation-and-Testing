#!/usr/bin/env python3
name = "6h_RSI_Bollinger_Band_Width_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Bollinger Bands and RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI(14)
    delta = np.diff(df_1d['close'])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(df_1d['close'])
    avg_loss = np.zeros_like(df_1d['close'])
    avg_gain[14] = np.mean(gain[:14])
    avg_loss[14] = np.mean(loss[:14])
    for i in range(15, len(df_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:14] = np.nan
    
    # Calculate 1d Bollinger Bands (20, 2)
    sma_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20 * 100  # Band width as percentage
    
    # Align 1d indicators to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Calculate 6h Bollinger Bands for entry signals
    sma_6h = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_6h = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb_6h = sma_6h + 2 * std_6h
    lower_bb_6h = sma_6h - 2 * std_6h
    
    # Position sizing
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(bb_width_aligned[i]) or
            np.isnan(upper_bb_6h[i]) or np.isnan(lower_bb_6h[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: Bollinger Band Width regime detection
        # Low volatility regime (BBW < 20th percentile) = mean reversion
        # High volatility regime (BBW > 80th percentile) = trend following
        bbw_low_threshold = np.nanpercentile(bb_width_aligned[:i+1], 20)
        bbw_high_threshold = np.nanpercentile(bb_width_aligned[:i+1], 80)
        
        low_vol_regime = bb_width_aligned[i] < bbw_low_threshold
        high_vol_regime = bb_width_aligned[i] > bbw_high_threshold
        
        # Mean reversion signals in low volatility regime
        if low_vol_regime:
            # RSI oversold bounce
            rsi_oversold = rsi_1d_aligned[i] < 30
            # Price touches lower Bollinger Band
            price_at_lower = close[i] <= lower_bb_6h[i]
            
            if position == 0 and rsi_oversold and price_at_lower:
                signals[i] = position_size
                position = 1
            elif position == 1 and (rsi_1d_aligned[i] > 50 or close[i] >= sma_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size if position == 1 else 0.0
        
        # Trend following signals in high volatility regime
        elif high_vol_regime:
            # RSI overbought continuation
            rsi_overbought = rsi_1d_aligned[i] > 70
            # Price touches upper Bollinger Band
            price_at_upper = close[i] >= upper_bb_6h[i]
            
            if position == 0 and rsi_overbought and price_at_upper:
                signals[i] = position_size
                position = 1
            elif position == 1 and (rsi_1d_aligned[i] < 50 or close[i] <= sma_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size if position == 1 else 0.0
        
        # Neutral regime: no action
        else:
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals