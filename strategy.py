#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla Pivot + Volume Spike
# - Uses 1d data to calculate weekly Camarilla pivot levels (H3/L3 for fade, H4/L4 for breakout)
# - Enters long at L3 with volume spike (>2x 20-period volume SMA), exits at H3
# - Enters short at H3 with volume spike, exits at L3
# - Breakout continuation: if price closes above H4 (or below L4) with volume, holds until opposite H3/L3
# - Weekly pivots calculated from prior week's H/L/C, aligned to 6h timeframe
# - Target: 20-40 trades/year to minimize fee drag while capturing institutional levels
# - Works in both bull/bear: mean reversion at H3/L3 in range, breakout continuation in trends

name = "6h_1w_camarilla_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for weekly Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate weekly Camarilla pivot levels from prior week's data
    # Camarilla formula: 
    # H4 = Close + 1.1*(High - Low)
    # H3 = Close + 1.1*(High - Low)/2
    # L3 = Close - 1.1*(High - Low)/2
    # L4 = Close - 1.1*(High - Low)
    
    # We need to get weekly OHLC from daily data
    # Resample daily to weekly using pandas (done once outside loop)
    df_1d_indexed = df_1d.copy()
    df_1d_indexed.index = pd.to_datetime(df_1d_indexed['open_time'])
    weekly = df_1d_indexed.resample('W-FRI').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    if len(weekly) < 10:
        return signals
    
    # Calculate Camarilla levels for each weekly bar
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    camarilla_h4 = weekly_close + 1.1 * (weekly_high - weekly_low)
    camarilla_h3 = weekly_close + 1.1 * (weekly_high - weekly_low) / 2
    camarilla_l3 = weekly_close - 1.1 * (weekly_high - weekly_low) / 2
    camarilla_l4 = weekly_close - 1.1 * (weekly_high - weekly_low)
    
    # Align weekly Camarilla levels to 1d timeframe (forward fill from weekly close)
    # Create series indexed by weekly dates, then reindex to daily
    weekly_dates = weekly.index
    h4_series = pd.Series(camarilla_h4, index=weekly_dates)
    h3_series = pd.Series(camarilla_h3, index=weekly_dates)
    l3_series = pd.Series(camarilla_l3, index=weekly_dates)
    l4_series = pd.Series(camarilla_l4, index=weekly_dates)
    
    # Reindex to daily frequency, forward fill
    daily_index = pd.to_datetime(df_1d['open_time'])
    h4_daily = h4_series.reindex(daily_index, method='ffill').values
    h3_daily = h3_series.reindex(daily_index, method='ffill').values
    l3_daily = l3_series.reindex(daily_index, method='ffill').values
    l4_daily = l4_series.reindex(daily_index, method='ffill').values
    
    # Align Camarilla levels to 6h timeframe
    h3_6h = align_htf_to_ltf(prices, df_1d, h3_daily)
    l3_6h = align_htf_to_ltf(prices, df_1d, l3_daily)
    h4_6h = align_htf_to_ltf(prices, df_1d, h4_daily)
    l4_6h = align_htf_to_ltf(prices, df_1d, l4_daily)
    
    # Pre-compute 6h volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_6h[i]) or np.isnan(l3_6h[i]) or 
            np.isnan(h4_6h[i]) or np.isnan(l4_6h[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        # Price levels
        h3 = h3_6h[i]
        l3 = l3_6h[i]
        h4 = h4_6h[i]
        l4 = l4_6h[i]
        price = close[i]
        
        # Trading logic
        if position == 0:  # Flat - look for new entries
            # Mean reversion entries at H3/L3 with volume
            if price <= l3 and vol_confirm:
                position = 1
                entry_price = price
                signals[i] = 0.25
            elif price >= h3 and vol_confirm:
                position = -1
                entry_price = price
                signals[i] = -0.25
            # Breakout entries at H4/L4 with volume
            elif price >= h4 and vol_confirm:
                position = 1
                entry_price = price
                signals[i] = 0.25
            elif price <= l4 and vol_confirm:
                position = -1
                entry_price = price
                signals[i] = -0.25
        
        elif position == 1:  # Long position
            # Exit at H3 (mean reversion target) or if price closes below L4 (breakout failure)
            if price >= h3 or price < l4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        
        elif position == -1:  # Short position
            # Exit at L3 (mean reversion target) or if price closes above H4 (breakout failure)
            if price <= l3 or price > h4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
    
    return signals