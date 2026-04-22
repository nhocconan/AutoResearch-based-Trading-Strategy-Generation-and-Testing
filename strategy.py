# State your hypothesis
# Hypothesis: 4h price action near 1-day VWAP with volume confirmation and trend filter
# Uses 1-day VWAP as dynamic support/resistance and 1-day EMA50 for trend direction
# Long when price > VWAP + volume spike + uptrend; Short when price < VWAP + volume spike + downtrend
# Exit when price crosses back through VWAP
# VWAP acts as a fair value anchor, reducing false breakouts and improving win rate in both bull/bear markets

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for VWAP and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP components
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = typical_price_1d * volume_1d
    vwap_denominator = volume_1d
    
    # Cumulative sums for VWAP (reset daily)
    cum_vwap_num = np.cumsum(vwap_numerator)
    cum_vwap_den = np.cumsum(vwap_denominator)
    vwap_1d = cum_vwap_num / cum_vwap_den
    
    # Calculate 50-period EMA on daily close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50 = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 4-hour timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above VWAP + volume spike + uptrend (price > EMA50)
            if (close[i] > vwap_aligned[i] and vol_spike[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below VWAP + volume spike + downtrend (price < EMA50)
            elif (close[i] < vwap_aligned[i] and vol_spike[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back through VWAP
            if position == 1:
                if close[i] < vwap_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > vwap_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_VWAP_EMA50_Volume_Spike_Session"
timeframe = "4h"
leverage = 1.0