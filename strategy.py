#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d volume spike and trend filter
# Long when price touches Camarilla L3 support + volume spike + 1d uptrend
# Short when price touches Camarilla H3 resistance + volume spike + 1d downtrend
# Exit when price reaches Camarilla H4/L4 or trend reverses
# Designed for 15-30 trades/year on 12h timeframe with mean-reversion in ranging markets
# Works in both bull/bear via trend filter and mean-reversion logic

name = "12h_1d_camarilla_volume_trend_v1"
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
    
    # Calculate 1d EMA(20) for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 24-period average volume for volume filter (24*12h = 12d)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate Camarilla levels from previous 12h bar
    # Camarilla: H4 = C + 1.5*(H-L), H3 = C + 1.25*(H-L), L3 = C - 1.25*(H-L), L4 = C - 1.5*(H-L)
    camarilla_high = np.roll(high, 1)
    camarilla_low = np.roll(low, 1)
    camarilla_close = np.roll(close, 1)
    camarilla_high[0] = np.nan
    camarilla_low[0] = np.nan
    camarilla_close[0] = np.nan
    
    camarilla_range = camarilla_high - camarilla_low
    camarilla_h3 = camarilla_close + 1.25 * camarilla_range
    camarilla_l3 = camarilla_close - 1.25 * camarilla_range
    camarilla_h4 = camarilla_close + 1.5 * camarilla_range
    camarilla_l4 = camarilla_close - 1.5 * camarilla_range
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma_24[i]) or np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2.0x 24-period average
        volume_filter = volume[i] > 2.0 * vol_ma_24[i]
        
        # Trend filter: price relative to 1d EMA20
        is_uptrend = close[i] > ema_20_1d_aligned[i]
        is_downtrend = close[i] < ema_20_1d_aligned[i]
        
        # Entry conditions: price touches Camarilla H3/L3 with volume and trend
        long_entry = (close[i] <= camarilla_l3[i] and low[i] <= camarilla_l3[i]) and volume_filter and is_uptrend
        short_entry = (close[i] >= camarilla_h3[i] and high[i] >= camarilla_h3[i]) and volume_filter and is_downtrend
        
        # Exit conditions: price reaches H4/L4 or trend reverses
        long_exit = (close[i] >= camarilla_h4[i]) or (not is_uptrend)
        short_exit = (close[i] <= camarilla_l4[i]) or (not is_downtrend)
        
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