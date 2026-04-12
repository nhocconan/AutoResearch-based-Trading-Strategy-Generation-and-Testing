#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w trend filter + volume confirmation
    # Only long when 1w close > 1w EMA50 (bullish regime), short when below (bearish regime)
    # Enter on breakout of Camarilla H3/L3 levels with volume > 2.0 * 20-period average
    # Exit on opposite Camarilla level (L3 for longs, H3 for shorts) or trend reversal
    # Discrete sizing 0.25 to minimize fee churn. Target: 20-40 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Previous day's high, low, close
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        rang = phigh - plow
        
        # Camarilla levels
        camarilla_h3[i] = pclose + rang * 1.1 / 4
        camarilla_l3[i] = pclose - rang * 1.1 / 4
        camarilla_h4[i] = pclose + rang * 1.1 / 2
        camarilla_l4[i] = pclose - rang * 1.1 / 2
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1w trend
        bullish_trend = close[i] > ema50_1w_aligned[i]
        bearish_trend = close[i] < ema50_1w_aligned[i]
        
        # Entry logic: Camarilla breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above Camarilla H3 in bullish trend
        if bullish_trend:
            long_entry = (close[i] > camarilla_h3[i]) and volume_spike[i]
        # Short breakout: price breaks below Camarilla L3 in bearish trend
        elif bearish_trend:
            short_entry = (close[i] < camarilla_l3[i]) and volume_spike[i]
        
        # Exit logic: opposite Camarilla level or trend reversal
        long_exit = (bearish_trend and close[i] < camarilla_l3[i]) or (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > camarilla_h3[i]) or (not bullish_trend and not bearish_trend)
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_breakout_trend_volume_v1"
timeframe = "1d"
leverage = 1.0