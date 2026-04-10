#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above H3 Camarilla level AND 4h close > 4h EMA(50) AND volume > 1.5x 20-period average volume
# - Short when price breaks below L3 Camarilla level AND 4h close < 4h EMA(50) AND volume > 1.5x 20-period average volume
# - Exit when price crosses back inside H3/L3 levels or 4h trend reverses
# - Uses discrete position sizing 0.20 to limit fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Camarilla pivots provide precise intraday support/resistance levels
# - 4h EMA filter ensures we trade with the higher timeframe trend
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
    
    # Pre-compute 1h Camarilla pivot levels (based on previous day)
    # Calculate daily pivot points from 1d data (we'll use 4h as proxy for daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC for Camarilla calculation
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Align previous day's data to 1h timeframe
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
    prev_day_close_aligned = align_htf_to_ltf(prices, df_1d, prev_day_close)
    
    # Calculate Camarilla levels
    # H4 = Close + 1.5*(High-Low), H3 = Close + 1.0*(High-Low), etc.
    # L3 = Close - 1.0*(High-Low), L4 = Close - 1.5*(High-Low)
    camarilla_range = prev_day_high_aligned - prev_day_low_aligned
    h3 = prev_day_close_aligned + 1.0 * camarilla_range
    l3 = prev_day_close_aligned - 1.0 * camarilla_range
    h4 = prev_day_close_aligned + 1.5 * camarilla_range
    l4 = prev_day_close_aligned - 1.5 * camarilla_range
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Pre-compute 1h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(ema_4h_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND 4h uptrend AND volume spike
            if (close[i] > h3[i] and 
                close[i] > ema_4h_50_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below L3 AND 4h downtrend AND volume spike
            elif (close[i] < l3[i] and 
                  close[i] < ema_4h_50_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside H3/L3 levels OR 4h trend reverses
            exit_long = (position == 1 and (close[i] < h3[i] or close[i] < ema_4h_50_aligned[i]))
            exit_short = (position == -1 and (close[i] > l3[i] or close[i] > ema_4h_50_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals