#!/usr/bin/env python3
"""
4h_cci_breakout_12h_trend_volume_v6
Hypothesis: On 4h timeframe, use 12h CCI for trend strength and direction, with 12h EMA for trend filter, and volume confirmation for institutional participation. Enter long when CCI crosses above +100 with price above EMA and volume confirmation; enter short when CCI crosses below -100 with price below EMA and volume confirmation. Exit when CCI returns to zero or opposite extreme. This strategy targets strong trending moves with volume confirmation, reducing false signals and trade frequency. Works in bull/bear via trend filter and breakout logic. Adjusted to reduce trade frequency and improve robustness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_breakout_12h_trend_volume_v6"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for CCI and EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate CCI on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price
    tp_12h = (high_12h + low_12h + close_12h) / 3
    # SMA of typical price
    sma_tp = pd.Series(tp_12h).rolling(window=20, min_periods=20).mean().values
    # Mean deviation
    md = pd.Series(tp_12h).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # CCI calculation
    cci_12h = (tp_12h - sma_tp) / (0.015 * md)
    
    # 12h EMA for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    
    # Align indicators to 4h timeframe
    cci_12h_4h = align_htf_to_ltf(prices, df_12h, cci_12h)
    ema_12h_4h = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(cci_12h_4h[i]) or np.isnan(ema_12h_4h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (increased threshold)
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend direction from EMA
        uptrend = close[i] > ema_12h_4h[i]
        downtrend = close[i] < ema_12h_4h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if CCI returns to zero (trend weakening)
            if abs(cci_12h_4h[i]) < 10:
                exit_long = True
            # Exit if CCI goes below -100 (strong reversal)
            elif cci_12h_4h[i] < -100:
                exit_long = True
            # Exit if trend turns down
            elif downtrend and cci_12h_4h[i] < 0:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if CCI returns to zero (trend weakening)
            if abs(cci_12h_4h[i]) < 10:
                exit_short = True
            # Exit if CCI goes above +100 (strong reversal)
            elif cci_12h_4h[i] > 100:
                exit_short = True
            # Exit if trend turns up
            elif uptrend and cci_12h_4h[i] > 0:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # CCI breaks above +100 with uptrend and volume confirmation
            if cci_12h_4h[i] > 100 and cci_12h_4h[i-1] <= 100:
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # CCI breaks below -100 with downtrend and volume confirmation
            if cci_12h_4h[i] < -100 and cci_12h_4h[i-1] >= -100:
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals