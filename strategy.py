#!/usr/bin/env python3
# 6h_1d_cci_trend_reversal_v1
# Hypothesis: 6-hour CCI trend reversal with 1-day trend filter and volume confirmation.
# Long when CCI crosses above -100 with price above 1-day EMA50 and volume > 1.3x 20-period average.
# Short when CCI crosses below +100 with price below 1-day EMA50 and volume > 1.3x 20-period average.
# Uses CCI for mean reversion in trends, filtered by higher timeframe trend to avoid counter-trend trades.
# Volume confirmation reduces false signals. Designed for 15-30 trades/year on 6h.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_trend_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # CCI(20) calculation
    typical_price = (high + low + close) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    # Handle division by zero or near-zero mad
    cci = np.where(mad == 0, 0, cci)
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 6h volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 50)  # CCI(20) and EMA50 need 20 and 50 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(cci[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = vol_ma_20[i] > 0 and volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: CCI crosses below +100 (overbought reversal) or trend breaks
            if cci[i] < 100 and cci[i-1] >= 100:
                position = 0
                signals[i] = 0.0
            elif close[i] < ema50_1d_aligned[i]:  # Trend break
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above -100 (oversold reversal) or trend breaks
            if cci[i] > -100 and cci[i-1] <= -100:
                position = 0
                signals[i] = 0.0
            elif close[i] > ema50_1d_aligned[i]:  # Trend break
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI crosses above -100 with uptrend and volume
            if cci[i] > -100 and cci[i-1] <= -100 and close[i] > ema50_1d_aligned[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: CCI crosses below +100 with downtrend and volume
            elif cci[i] < 100 and cci[i-1] >= 100 and close[i] < ema50_1d_aligned[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals