#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Uses 4h EMA(50) for trend direction (long only when price > EMA50, short when price < EMA50)
# - 1h Camarilla pivot levels (H3/L3) as breakout triggers with volume > 1.5x 20-bar average
# - Long when price breaks above H3 with volume spike and 4h uptrend
# - Short when price breaks below L3 with volume spike and 4h downtrend
# - Exit when price returns to pivot point (PP) or opposite Camarilla level is reached
# - Discrete position sizing (0.20) to minimize fee churn
# - Targets 15-35 trades/year (60-140 total over 4 years) to avoid fee drag
# - Camarilla pivots work well in ranging markets and catch breakouts in trends
# - Volume confirmation filters false breakouts
# - 4h trend filter ensures alignment with higher timeframe momentum
# - Session filter (08-20 UTC) reduces noise during low-activity periods

name = "1h_4h_camarilla_pivot_breakout_volume_trend_v1"
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
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1h volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if outside trading session (08-20 UTC)
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        # Calculate 1h Camarilla pivot levels for current bar
        high = prices['high'].iloc[i-1]  # Use previous bar's data for pivot calculation
        low = prices['low'].iloc[i-1]
        close = prices['close'].iloc[i-1]
        
        # Camarilla pivot calculations
        range_val = high - low
        if range_val <= 0:
            # Skip if invalid range
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
            
        pp = (high + low + close) / 3
        h3 = pp + (range_val * 1.1 / 4)
        l3 = pp - (range_val * 1.1 / 4)
        h4 = pp + (range_val * 1.1 / 2)
        l4 = pp - (range_val * 1.1 / 2)
        
        current_price = prices['close'].iloc[i]
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above H3 with volume spike and 4h uptrend
            if (current_price > h3 and 
                vol_spike.iloc[i] and 
                current_price > ema_50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short signal: price breaks below L3 with volume spike and 4h downtrend
            elif (current_price < l3 and 
                  vol_spike.iloc[i] and 
                  current_price < ema_50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to pivot point (PP)
            # 2. Price reaches opposite extreme (H4 for longs, L4 for shorts)
            if position == 1:  # Long position
                if current_price <= pp or current_price >= h4:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20  # Hold long
            elif position == -1:  # Short position
                if current_price >= pp or current_price <= l4:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20  # Hold short
    
    return signals