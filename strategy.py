#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with volume confirmation and 4h trend filter
# Uses Camarilla pivot levels (H3/L3) from 4h for structure, breaks above/below for entries
# Only takes breakouts when 4h EMA(50) > EMA(200) for uptrend (longs) or EMA(50) < EMA(200) for downtrend (shorts)
# Volume confirmation ensures breakouts have participation
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag
# Session filter (08-20 UTC) reduces noise trades
# Works in both bull/bear: 4h trend filter ensures we trade with higher timeframe momentum

name = "1h_4h_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Camarilla pivots and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels (H3, L3) using previous bar's data to avoid look-ahead
    camarilla_h3 = np.full(len(df_4h), np.nan)
    camarilla_l3 = np.full(len(df_4h), np.nan)
    camarilla_h4 = np.full(len(df_4h), np.nan)
    camarilla_l4 = np.full(len(df_4h), np.nan)
    
    for i in range(len(df_4h)):
        if i < 1:
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
        else:
            # Use previous 4h bar's OHLC to calculate current pivot levels
            prev_high = df_4h['high'].iloc[i-1]
            prev_low = df_4h['low'].iloc[i-1]
            prev_close = df_4h['close'].iloc[i-1]
            
            pivot = (prev_high + prev_low + prev_close) / 3.0
            range_val = prev_high - prev_low
            
            camarilla_h3[i] = pivot + range_val * 1.1 / 4.0
            camarilla_l3[i] = pivot - range_val * 1.1 / 4.0
            camarilla_h4[i] = pivot + range_val * 1.1 / 2.0
            camarilla_l4[i] = pivot - range_val * 1.1 / 2.0
    
    # Calculate 4h EMA(50) and EMA(200) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h Camarilla levels and EMAs to 1h timeframe
    camarilla_h3_1h = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_1h = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_h4_1h = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_1h = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    ema_50_4h_1h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_1h = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3_1h[i]) or 
            np.isnan(camarilla_l3_1h[i]) or 
            np.isnan(ema_50_4h_1h[i]) or 
            np.isnan(ema_200_4h_1h[i]) or 
            np.isnan(avg_volume[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        # Trend filter: EMA(50) > EMA(200) for uptrend, EMA(50) < EMA(200) for downtrend
        uptrend = ema_50_4h_1h[i] > ema_200_4h_1h[i]
        downtrend = ema_50_4h_1h[i] < ema_200_4h_1h[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Camarilla L3 OR trend turns down
            if close[i] < camarilla_l3_1h[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Camarilla H3 OR trend turns up
            if close[i] > camarilla_h3_1h[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic: Camarilla breakout with volume confirmation and trend filter
            if volume_confirm:
                # Long breakout: price closes above Camarilla H3 in uptrend
                if close[i] > camarilla_h3_1h[i] and uptrend:
                    position = 1
                    signals[i] = 0.20
                # Short breakout: price closes below Camarilla L3 in downtrend
                elif close[i] < camarilla_l3_1h[i] and downtrend:
                    position = -1
                    signals[i] = -0.20
    
    return signals