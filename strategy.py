#!/usr/bin/env python3
# 6h_12h_1d_cci_trend_reversal_v1
# Hypothesis: Trend reversal using CCI on 6h with 12h trend filter and volume confirmation.
# In bull markets: long when CCI crosses above -100 from below (bullish reversal) with volume surge and 12h uptrend.
# In bear markets: short when CCI crosses below +100 from above (bearish reversal) with volume surge and 12h downtrend.
# CCI captures overbought/oversold conditions and mean reversion, while higher timeframe filter ensures alignment with larger trend.
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_cci_trend_reversal_v1"
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
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mean_dev = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = np.zeros_like(typical_price)
    cci[20:] = (typical_price[20:] - sma_tp[20:]) / (0.015 * mean_dev[20:])
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h EMA21 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema21_12h = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 40  # Ensure all indicators are ready (CCI needs 20 + buffer)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cci[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema21_12h_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # CCI crossover signals
        cci_cross_up = cci[i] > -100 and cci[i-1] <= -100  # Cross above -100 (bullish)
        cci_cross_down = cci[i] < 100 and cci[i-1] >= 100  # Cross below +100 (bearish)
        
        if position == 1:  # Long position
            # Exit: CCI crosses below +100 (overbought) or trend breaks (price < 12h EMA21)
            if cci[i] < 100 and cci[i-1] >= 100 or close[i] < ema21_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above -100 (oversold) or trend breaks (price > 12h EMA21)
            if cci[i] > -100 and cci[i-1] <= -100 or close[i] > ema21_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI crosses above -100 with volume surge and 12h uptrend
            if cci_cross_up and vol_surge and close[i] > ema21_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: CCI crosses below +100 with volume surge and 12h downtrend
            elif cci_cross_down and vol_surge and close[i] < ema21_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals