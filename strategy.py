#!/usr/bin/env python3
"""
1d_cci_breakout_weekly_trend_volume_v1
Hypothesis: On daily timeframe, use weekly CCI for trend strength and direction, with weekly EMA for trend filter, and volume confirmation for institutional participation. Enter long when CCI crosses above +100 with price above EMA and volume confirmation; enter short when CCI crosses below -100 with price below EMA and volume confirmation. Exit when CCI returns to zero or opposite extreme. This strategy targets strong trending moves with volume confirmation, reducing false signals and trade frequency. Works in bull/bear via trend filter and breakout logic. Adjusted to reduce trade frequency by using weekly trend filter and daily entries, targeting 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_cci_breakout_weekly_trend_volume_v1"
timeframe = "1d"
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
    
    # Weekly data for CCI and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate CCI on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price
    tp_1w = (high_1w + low_1w + close_1w) / 3
    # SMA of typical price
    sma_tp = pd.Series(tp_1w).rolling(window=20, min_periods=20).mean().values
    # Mean deviation
    md = pd.Series(tp_1w).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # CCI calculation
    cci_1w = (tp_1w - sma_tp) / (0.015 * md)
    
    # Weekly EMA for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    
    # Align indicators to daily timeframe
    cci_1w_1d = align_htf_to_ltf(prices, df_1w, cci_1w)
    ema_1w_1d = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation (20-period average on daily)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(cci_1w_1d[i]) or np.isnan(ema_1w_1d[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend direction from EMA
        uptrend = close[i] > ema_1w_1d[i]
        downtrend = close[i] < ema_1w_1d[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if CCI returns to zero (trend weakening)
            if abs(cci_1w_1d[i]) < 10:
                exit_long = True
            # Exit if CCI goes below -100 (strong reversal)
            elif cci_1w_1d[i] < -100:
                exit_long = True
            # Exit if trend turns down
            elif downtrend and cci_1w_1d[i] < 0:
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
            if abs(cci_1w_1d[i]) < 10:
                exit_short = True
            # Exit if CCI goes above +100 (strong reversal)
            elif cci_1w_1d[i] > 100:
                exit_short = True
            # Exit if trend turns up
            elif uptrend and cci_1w_1d[i] > 0:
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
            if cci_1w_1d[i] > 100 and cci_1w_1d[i-1] <= 100:
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # CCI breaks below -100 with downtrend and volume confirmation
            if cci_1w_1d[i] < -100 and cci_1w_1d[i-1] >= -100:
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals