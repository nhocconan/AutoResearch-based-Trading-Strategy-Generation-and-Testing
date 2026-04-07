#!/usr/bin/env python3
"""
6h_momentum_reversal_1d_trend_volume_v1
Hypothesis: On 6-hour timeframe, combine 1-day momentum (RSI) with price reversal near Bollinger Bands (20,2) and volume confirmation. In ranging markets, fade extremes; in trending markets (1-day RSI >50 or <50), continue momentum. Volume ensures institutional participation. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag while adapting to bull/bear regimes via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "6h_momentum_reversal_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands and RSI
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Bollinger Bands (20,2)
    close_1d = df_1d['close'].values
    bb_mid = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Calculate daily RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Align daily indicators to 6h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_mid_aligned = align_htf_to_ltf(prices, df_1d, bb_mid)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume filter: 24-period average on 6h timeframe (~6 days)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(24, 20), n):
        # Skip if data not available
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches middle band (take profit) or breaks below lower band (stop)
            if close[i] >= bb_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif low[i] <= bb_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches middle band (take profit) or breaks above upper band (stop)
            if close[i] <= bb_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif high[i] >= bb_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Mean reversion: fade at Bollinger Bands when RSI extreme
                # Long: price touches/below lower band with RSI < 30 (oversold)
                if low[i] <= bb_lower_aligned[i] and rsi_aligned[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches/above upper band with RSI > 70 (overbought)
                elif high[i] >= bb_upper_aligned[i] and rsi_aligned[i] > 70:
                    position = -1
                    signals[i] = -0.25
                # Momentum continuation: break through bands with aligned RSI trend
                # Long: break above upper band with RSI > 50 (bullish momentum)
                elif high[i] >= bb_upper_aligned[i] and rsi_aligned[i] > 50:
                    position = 1
                    signals[i] = 0.25
                # Short: break below lower band with RSI < 50 (bearish momentum)
                elif low[i] <= bb_lower_aligned[i] and rsi_aligned[i] < 50:
                    position = -1
                    signals[i] = -0.25
    
    return signals