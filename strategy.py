#!/usr/bin/env python3
# 4h_cci_trend_reversal_v1
# Hypothesis: 4h strategy using CCI(20) for overbought/oversold conditions with trend filter from 12h EMA(50) and volume confirmation.
# In bull markets: buy oversold dips in uptrend; in bear markets: sell overbought rallies in downtrend.
# Volume confirmation filters weak moves. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_trend_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # CCI(20) calculation
    tp = (high + low + close) / 3.0  # Typical price
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    md_tp = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    md_tp = np.where(md_tp == 0, 1e-10, md_tp)
    cci = (tp - ma_tp) / (0.015 * md_tp)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: CCI crosses above +100 (overbought) or volume dries up
            if cci[i] > 100.0 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses below -100 (oversold) or volume dries up
            if cci[i] < -100.0 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: CCI crosses above -100 from below (oversold bounce) AND uptrend filter (price > 12h EMA50)
                if cci[i] > -100.0 and cci[i-1] <= -100.0 and close[i] > ema_50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: CCI crosses below +100 from above (overbought rejection) AND downtrend filter (price < 12h EMA50)
                elif cci[i] < 100.0 and cci[i-1] >= 100.0 and close[i] < ema_50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals