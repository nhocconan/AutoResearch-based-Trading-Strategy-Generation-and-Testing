#!/usr/bin/env python3
"""
1d Bollinger Band Width Regime + RSI Mean Reversion with Volume Confirmation
Trades Bollinger Band squeezes in low volatility regimes with RSI extremes.
Designed for low trade frequency and works in both bull and bear markets.
"""

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
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Bollinger Band Width (normalized)
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # RSI (14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align to 1d timeframe (no additional delay needed for BB width and RSI)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # Volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(bb_width_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_width_val = bb_width_aligned[i]
        rsi_val = rsi_aligned[i]
        sma_val = sma_20_aligned[i]
        
        # Regime filter: low volatility (BB width < 20th percentile lookback)
        if i >= 60:
            bb_width_lookback = bb_width_aligned[max(0, i-60):i]
            bb_width_percentile = np.percentile(bb_width_lookback, 20) if len(bb_width_lookback) > 0 else 0.1
            low_vol_regime = bb_width_val < bb_width_percentile
        else:
            low_vol_regime = False
        
        if position == 0:
            # Long: RSI oversold (<30) + low volatility regime + price near lower BB + volume spike
            if (rsi_val < 30 and 
                low_vol_regime and 
                price <= sma_val and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + low volatility regime + price near upper BB + volume spike
            elif (rsi_val > 70 and 
                  low_vol_regime and 
                  price >= sma_val and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 or price > upper BB
            if rsi_val > 50 or price >= upper_bb[np.sum(df_1d['close'].values <= price) if len(df_1d['close'].values) > 0 else 0]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 or price < lower BB
            if rsi_val < 50 or price <= lower_bb[np.sum(df_1d['close'].values <= price) if len(df_1d['close'].values) > 0 else 0]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_BBWidthRegime_RSI_MeanRev_Volume"
timeframe = "1d"
leverage = 1.0