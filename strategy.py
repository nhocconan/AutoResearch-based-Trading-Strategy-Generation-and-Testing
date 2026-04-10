#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with volume confirmation and 1d trend filter
# - Long when price breaks above Camarilla R4 with volume > 1.5x 20-period average AND price > 1d EMA(50)
# - Short when price breaks below Camarilla S4 with volume > 1.5x 20-period average AND price < 1d EMA(50)
# - Exit when price returns to Camarilla R3/S3 levels or opposite signal
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Camarilla levels provide institutional support/resistance; volume confirms institutional participation
# - 1d EMA filter ensures alignment with higher timeframe trend, reducing whipsaws in ranging markets

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h Camarilla pivot levels from previous day
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    # For each 6h bar, use the prior 1d bar's OHLC (4x6h = 1d)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(4, n):  # Need at least 4x6h bars for 1d lookback
        # Get prior 1d bar's OHLC (indices i-4 to i-1 for 6h bars)
        idx_start = i - 4
        idx_end = i  # exclusive
        
        # Calculate daily OHLC from 4x6h bars
        day_high = np.max(high[idx_start:idx_end])
        day_low = np.min(low[idx_start:idx_end])
        day_close = close[idx_end - 1]  # close of last 6h bar in the day
        
        # Camarilla formula
        range_val = day_high - day_low
        camarilla_h4[i] = day_close + range_val * 1.1 / 2
        camarilla_l4[i] = day_close - range_val * 1.1 / 2
        camarilla_h3[i] = day_close + range_val * 1.1 / 4
        camarilla_l3[i] = day_close - range_val * 1.1 / 4
    
    # Pre-compute 6h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Pre-compute 1d EMA(50) trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above R4 with volume spike AND price > 1d EMA(50)
            if (close[i] > camarilla_h4[i] and volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below S4 with volume spike AND price < 1d EMA(50)
            elif (close[i] < camarilla_l4[i] and volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when price returns to R3 or opposite signal
            exit_long = (position == 1 and 
                        (close[i] < camarilla_h3[i] or  # price returned to R3
                         close[i] < camarilla_l4[i]))   # or broke below S4 (stop)
            # Exit short when price returns to S3 or opposite signal
            exit_short = (position == -1 and 
                         (close[i] > camarilla_l3[i] or  # price returned to S3
                          close[i] > camarilla_h4[i]))  # or broke above R4 (stop)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals