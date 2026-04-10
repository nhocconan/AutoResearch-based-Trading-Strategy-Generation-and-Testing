#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 1w EMA(21) > EMA(50) AND volume > 1.5x 20-period average
# - Short when price breaks below Camarilla L3 level AND 1w EMA(21) < EMA(50) AND volume > 1.5x 20-period average
# - Exit when price crosses back inside H3/L3 levels
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance levels
# - 1w EMA filter ensures we trade with the weekly trend
# - Volume confirmation reduces false breakouts

name = "12h_1w_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h Camarilla pivot levels (based on previous day)
    def calculate_camarilla(h_prev, l_prev, c_prev):
        range_val = h_prev - l_prev
        if range_val <= 0:
            return c_prev, c_prev, c_prev, c_prev
        h3 = c_prev + (range_val * 1.1 / 4)
        l3 = c_prev - (range_val * 1.1 / 4)
        h4 = c_prev + (range_val * 1.1 / 2)
        l4 = c_prev - (range_val * 1.1 / 2)
        return h3, l3, h4, l4
    
    # Calculate daily Camarilla levels
    h3_1d = np.full(len(df_1d), np.nan)
    l3_1d = np.full(len(df_1d), np.nan)
    h4_1d = np.full(len(df_1d), np.nan)
    l4_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        h3, l3, h4, l4 = calculate_camarilla(
            df_1d['high'].iloc[i-1],
            df_1d['low'].iloc[i-1],
            df_1d['close'].iloc[i-1]
        )
        h3_1d[i] = h3
        l3_1d[i] = l3
        h4_1d[i] = h4
        l4_1d[i] = l4
    
    # Align daily Camarilla levels to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w EMA trend filter
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = ema_21 > ema_50
    downtrend = ema_21 < ema_50
    
    # Align HTF indicators to 12h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1w, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1w, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(uptrend_aligned[i]) or 
            np.isnan(downtrend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND weekly uptrend AND volume spike
            if (close[i] > h3_1d_aligned[i] and 
                uptrend_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND weekly downtrend AND volume spike
            elif (close[i] < l3_1d_aligned[i] and 
                  downtrend_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside H3/L3 levels
            exit_long = (position == 1 and close[i] < h3_1d_aligned[i])
            exit_short = (position == -1 and close[i] > l3_1d_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals