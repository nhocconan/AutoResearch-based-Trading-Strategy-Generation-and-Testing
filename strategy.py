#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 4h close > 4h EMA50 (uptrend) AND volume > 1.5x 20-period average volume
# - Short when price breaks below Camarilla L3 level AND 4h close < 4h EMA50 (downtrend) AND volume > 1.5x 20-period average volume
# - Exit when price crosses back to Camarilla pivot point (central level)
# - Uses discrete position sizing 0.20 to limit fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Camarilla pivots provide intraday support/resistance levels effective in both trending and ranging markets
# - 4h EMA50 filter ensures we trade with higher timeframe trend to avoid counter-trend whipsaws
# - Volume confirmation reduces false breakouts

name = "1h_4h_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 1h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1h Camarilla pivots (based on previous day's OHLC)
    # We'll calculate daily pivots first, then align to 1h
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day: based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, H2, H1, Pivot, L1, L2, L3, L4
    # H3 = Close + 1.1*(High-Low)/2
    # L3 = Close - 1.1*(High-Low)/2
    # Pivot = (High + Low + Close)/3
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices.iloc[i]['open_time']).hour
        if hour < 8 or hour > 20:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND 4h uptrend AND volume spike
            if (close[i] > camarilla_h3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and  # Price above 4h EMA50 (uptrend confirmation)
                volume_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below Camarilla L3 AND 4h downtrend AND volume spike
            elif (close[i] < camarilla_l3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and  # Price below 4h EMA50 (downtrend confirmation)
                  volume_spike[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back to Camarilla pivot point
            exit_long = (position == 1 and close[i] < camarilla_pivot_aligned[i])
            exit_short = (position == -1 and close[i] > camarilla_pivot_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals