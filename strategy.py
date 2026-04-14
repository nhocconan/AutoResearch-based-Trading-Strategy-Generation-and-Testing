#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot + 1d Volume Spike + 1w EMA Trend Filter
# Uses Camarilla pivot levels from daily data with volume confirmation and weekly trend filter
# Weekly EMA(50) ensures we only trade in the direction of the higher timeframe trend
# Works in bull/bear by trading mean reversion to pivot levels in trending markets
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla levels use previous day's high, low, close
    phigh_1d = df_1d['high'].values
    plow_1d = df_1d['low'].values
    pclose_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h4 = np.zeros(len(df_1d))  # Resistance level
    camarilla_l4 = np.zeros(len(df_1d))  # Support level
    camarilla_h3 = np.zeros(len(df_1d))
    camarilla_l3 = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        # Previous day's values
        phigh = phigh_1d[i-1]
        plow = plow_1d[i-1]
        pclose = pclose_1d[i-1]
        
        range_val = phigh - plow
        camarilla_h4[i] = pclose + range_val * 1.1 / 2
        camarilla_l4[i] = pclose - range_val * 1.1 / 2
        camarilla_h3[i] = pclose + range_val * 1.1 / 4
        camarilla_l3[i] = pclose - range_val * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 1d volume for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=10, min_periods=10).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_ma_1d)  # Volume > 2x average
    vol_spike_12h = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Get 1w EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for 1w EMA calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or
            np.isnan(ema_50_12h[i]) or np.isnan(vol_spike_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_spike_12h[i] > 0.5
        
        # Trend filter: only trade when price is above/below weekly EMA
        if ema_50_12h[i] <= 0:  # Invalid EMA
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price drops to L3/L4 with volume spike in uptrend
            if price <= l4_12h[i] and vol_spike and price > ema_50_12h[i]:
                position = 1
                signals[i] = position_size
            # Short: price rises to H3/H4 with volume spike in downtrend
            elif price >= h4_12h[i] and vol_spike and price < ema_50_12h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 or H4
            if price >= h3_12h[i] or price >= h4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 or L4
            if price <= l3_12h[i] or price <= l4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_1dVolumeSpike_1wEMA_Trend"
timeframe = "12h"
leverage = 1.0