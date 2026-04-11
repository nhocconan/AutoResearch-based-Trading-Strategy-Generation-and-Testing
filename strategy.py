#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day volume-weighted average price (VWAP) deviation + volume confirmation.
# Long when price deviates >1.5σ below VWAP with volume spike, short when >1.5σ above VWAP with volume spike.
# Uses statistical deviation from VWAP as mean-reversion signal, avoiding overtrading via strict volatility filter.
# Designed for 20-40 trades/year to minimize fee drag while capturing mean-reversion in ranging markets.
# Works in both bull/bear markets as VWAP deviation signals exhaustion moves.

name = "4h_1d_vwap_deviation_volume_v1"
timeframe = "4h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily VWAP
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    tpv_1d = typical_price_1d * df_1d['volume']
    cum_tpv_1d = tpv_1d.cumsum()
    cum_vol_1d = df_1d['volume'].cumsum()
    vwap_1d = cum_tpv_1d / cum_vol_1d
    vwap_1d = vwap_1d.values
    
    # Calculate daily VWAP deviation standard deviation (20-period)
    deviation_1d = typical_price_1d - vwap_1d
    vol_dev_20_1d = pd.Series(deviation_1d).rolling(window=20, min_periods=20).std().values
    
    # Calculate daily average volume (20-period)
    vol_avg_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    vol_dev_aligned = align_htf_to_ltf(prices, df_1d, vol_dev_20_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure indicators are valid
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_aligned[i]) or np.isnan(vol_dev_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Avoid division by zero
        if vol_dev_aligned[i] <= 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate VWAP deviation in standard deviations
        dev_std = (close[i] - vwap_aligned[i]) / vol_dev_aligned[i]
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Entry conditions: price deviates >1.5σ from VWAP with volume spike
        long_entry = (dev_std < -1.5) and vol_filter
        short_entry = (dev_std > 1.5) and vol_filter
        
        # Exit conditions: price returns to within 0.5σ of VWAP
        exit_long = dev_std > -0.5
        exit_short = dev_std < 0.5
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals