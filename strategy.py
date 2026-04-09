#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with weekly trend filter and volume confirmation
# Camarilla levels provide precise intraday support/resistance derived from prior day's range
# Weekly trend filter (EMA34) ensures we trade with the higher timeframe momentum
# Volume confirmation (>1.5x 20-period average) filters false breakouts
# Works in bull/bear: weekly EMA adapts to trend direction, Camarilla provides structure in ranging markets
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d average volume (20-period) for confirmation
    volume_s = pd.Series(volume)
    avg_volume_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):  # Start after weekly EMA warmup
        # Skip if weekly EMA not available
        if np.isnan(ema_34_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Need prior day's OHLC for Camarilla calculation (i-1 must be valid)
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Prior day's OHLC
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla pivot levels for today (based on yesterday's range)
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        camarilla_h4 = prev_close + range_val * 1.1 / 2
        camarilla_l4 = prev_close - range_val * 1.1 / 2
        camarilla_h3 = prev_close + range_val * 1.1 / 4
        camarilla_l3 = prev_close - range_val * 1.1 / 4
        camarilla_h2 = prev_close + range_val * 1.1 / 6
        camarilla_l2 = prev_close - range_val * 1.1 / 6
        camarilla_h1 = prev_close + range_val * 1.1 / 12
        camarilla_l1 = prev_close - range_val * 1.1 / 12
        
        # Volume confirmation: current volume > 1.5 * 20-day average
        volume_confirmed = volume[i] > 1.5 * avg_volume_20[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR weekly trend turns down
            if close[i] < camarilla_l3 or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR weekly trend turns up
            if close[i] > camarilla_h3 or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: trade with weekly trend
            if weekly_uptrend:
                # Long on break above H3 or H4 with volume
                if (close[i] > camarilla_h3 or close[i] > camarilla_h4) and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
            elif weekly_downtrend:
                # Short on break below L3 or L4 with volume
                if (close[i] < camarilla_l3 or close[i] < camarilla_l4) and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals