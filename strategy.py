#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot + volume spike + session filter
# Camarilla levels (H3/L3) provide reversal zones in ranging markets and breakout confirmation in trends
# Long when price crosses above H3 with volume spike during active session (08-20 UTC)
# Short when price crosses below L3 with volume spike during active session
# Uses discrete position sizing 0.20 to target 15-30 trades/year and minimize fee drag
# Works in bull/bear markets: Camarilla adapts to volatility, session filter avoids low-liquidity periods

name = "1h_4h_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (H3, L3) based on previous day's range
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # Using rolling 4h window to approximate daily levels
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    # For 4h data, use 6-period lookback to approximate 1 day (6*4h = 24h)
    high_6 = rolling_max(high_4h, 6)
    low_6 = rolling_min(low_4h, 6)
    close_6 = pd.Series(close_4h).rolling(window=6, min_periods=6).mean().values
    
    camarilla_h3 = close_6 + 1.1 * (high_6 - low_6) / 4
    camarilla_l3 = close_6 - 1.1 * (high_6 - low_6) / 4
    
    # Align 4h indicators to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Pre-compute 1h volume moving average (20-period) for efficiency
    vol_s = pd.Series(volume)
    vol_ma_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 2.0x average 1h volume (20-period)
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit long if price falls below L3
            if close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit short if price rises above H3
            if close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Breakout strategy: enter on Camarilla level break with volume confirmation
            if close[i] > camarilla_h3_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.20
            elif close[i] < camarilla_l3_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.20
    
    return signals