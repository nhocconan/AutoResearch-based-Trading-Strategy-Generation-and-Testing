# 1h_4h1d_camarilla_volume_momentum_v1
# Hypothesis: 1h momentum strategy using 4h/1d Camarilla pivot levels for directional bias.
# Uses daily OHLC to calculate institutional-grade support/resistance (S4/R4).
# Breakouts beyond S4/R4 with volume confirmation capture institutional flow.
# Works in bull markets (buy R4 breakouts) and bear markets (sell S4 breakdowns).
# 1h timeframe for precise entry timing, 4h/1d for filtering false breakouts.
# Target: 15-37 trades/year (60-150 total over 4 years) with session filter (08-20 UTC).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_camarilla_volume_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots (1d HTF)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily high/low/close for Camarilla calculation
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate Camarilla pivot levels (S4/R4 only - outer bands for breakouts)
    # R4 = Close + (High - Low) * 1.1/2
    # S4 = Close - (High - Low) * 1.1/2
    camarilla_r4 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_s4 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Align daily Camarilla levels to 1h timeframe (shifted by 1 for completed bars)
    r4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s4)
    
    # Volume filter: volume > 2.0x 20-period average (strict for low frequency)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (avoid Asian session lows)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available or outside session
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below midpoint (take profit) or breaks S4 (stop)
            midpoint = (r4_aligned[i] + s4_aligned[i]) / 2
            if close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above midpoint (take profit) or breaks R4 (stop)
            midpoint = (r4_aligned[i] + s4_aligned[i]) / 2
            if close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above R4 with volume
            if (high[i] > r4_aligned[i] and close[i] > r4_aligned[i] and
                vol_filter[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: breakdown below S4 with volume
            elif (low[i] < s4_aligned[i] and close[i] < s4_aligned[i] and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals