#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot + 1d volume confirmation + session filter (08-20 UTC)
# - Primary: 4h Camarilla pivot levels (H3/L3) for mean reversion in ranging markets
# - HTF: 1d volume spike (current volume > 1.8x 20-period MA) for conviction
# - Session: Only trade during 08-20 UTC to avoid low-liquidity hours
# - Long: Price < L3 + volume confirmation + session active
# - Short: Price > H3 + volume confirmation + session active
# - Exit: Price crosses central pivot (PP) or session ends
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Works in bull/bear: Camarilla pivots adapt to volatility, volume confirms breakouts, session filter reduces noise
# - Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe

name = "1h_4h_1d_camarilla_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 5 or len(df_1d) < 5:  # Need enough data for indicators
        return np.zeros(n)
    
    # Pre-compute 1h data
    close_1h = prices['close'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    
    # Pre-compute 4h data for Camarilla pivots
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pre-compute 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Camarilla pivots (based on previous 4h bar)
    # Pivots calculated from previous bar's OHLC to avoid look-ahead
    pp_4h = np.full(len(close_4h), np.nan)
    h3_4h = np.full(len(close_4h), np.nan)
    l3_4h = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):  # Start from 1 to use previous bar
        if not (np.isnan(high_4h[i-1]) or np.isnan(low_4h[i-1]) or np.isnan(close_4h[i-1])):
            pp_4h[i] = (high_4h[i-1] + low_4h[i-1] + close_4h[i-1]) / 3.0
            range_4h = high_4h[i-1] - low_4h[i-1]
            h3_4h[i] = pp_4h[i] + range_4h * 1.1 / 4.0
            l3_4h[i] = pp_4h[i] - range_4h * 1.1 / 4.0
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all HTF indicators to 1h timeframe
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Pre-compute session filter (08-20 UTC)
    # open_time is already datetime64[ms], so we can use .hour directly
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_active = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from second bar to avoid index issues
        # Skip if any required data is invalid
        if (np.isnan(pp_4h_aligned[i]) or np.isnan(h3_4h_aligned[i]) or 
            np.isnan(l3_4h_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.8 * volume_ma_20_1d_aligned[i]
        
        # Session filter: only trade during 08-20 UTC
        in_session = session_active[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price < L3 + volume confirmation + session active
            if close_1h[i] < l3_4h_aligned[i] and volume_confirm and in_session:
                position = 1
                signals[i] = 0.20
            # Short entry: Price > H3 + volume confirmation + session active
            elif close_1h[i] > h3_4h_aligned[i] and volume_confirm and in_session:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses central pivot (PP) OR session ends
            if position == 1:  # Long position
                if close_1h[i] > pp_4h_aligned[i] or not in_session:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                if close_1h[i] < pp_4h_aligned[i] or not in_session:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals