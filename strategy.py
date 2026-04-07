#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Close Position with Volume Confirmation and ATR Filter
# Hypothesis: Price position relative to daily close captures institutional flow; volume confirms institutional participation; ATR filter avoids low-volatility whipsaws. Works in bull via trend continuation, in bear via mean reversion at extremes.
# Target: 20-50 trades/year to minimize fee drag.
name = "4h_daily_close_position_vol_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily close
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate daily volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ATR(14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(close_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Volatility filter: only trade when ATR is above its 50-period average
        atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else True
        
        # Price position relative to daily close
        price_pos = (close[i] - close_1d_aligned[i]) / close_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to daily close OR volatility drops
            if price_pos <= 0 or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price returns to daily close OR volatility drops
            if price_pos >= 0 or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter long: price above daily close + volume confirmation + volatility filter
            if price_pos > 0.01 and vol_confirm and vol_filter:  # 1% above daily close
                position = 1
                signals[i] = 0.25
            # Enter short: price below daily close + volume confirmation + volatility filter
            elif price_pos < -0.01 and vol_confirm and vol_filter:  # 1% below daily close
                position = -1
                signals[i] = -0.25
    
    return signals