#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d trend filter and volume confirmation
# Long when price touches Camarilla L3 support in uptrend with volume spike
# Short when price touches Camarilla H3 resistance in downtrend with volume spike
# Exit when price reaches Camarilla L4/H4 or trend reverses
# Designed for 12-37 trades/year on 12h timeframe with mean-reversion in trending markets

name = "12h_1d_camarilla_reversion_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous day
    # Use previous day's OHLC to avoid look-ahead
    prev_day_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_day_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_day_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Camarilla levels: 
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.1 * (High - Low)
    # L3 = Close - 1.1 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    camarilla_h4 = prev_day_close + 1.5 * (prev_day_high - prev_day_low)
    camarilla_h3 = prev_day_close + 1.1 * (prev_day_high - prev_day_low)
    camarilla_l3 = prev_day_close - 1.1 * (prev_day_high - prev_day_low)
    camarilla_l4 = prev_day_close - 1.5 * (prev_day_high - prev_day_low)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 to have previous day data
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: price touches Camarilla L3/H3 with volume and trend alignment
        # Long: price touches or goes below L3 in uptrend (mean reversion long)
        long_entry = (close[i] <= camarilla_l3_aligned[i]) and volume_filter and is_uptrend
        # Short: price touches or goes above H3 in downtrend (mean reversion short)
        short_entry = (close[i] >= camarilla_h3_aligned[i]) and volume_filter and is_downtrend
        
        # Exit conditions: price reaches L4/H4 or trend reverses
        long_exit = (close[i] >= camarilla_l4_aligned[i]) or (not is_uptrend)
        short_exit = (close[i] <= camarilla_h4_aligned[i]) or (not is_downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals