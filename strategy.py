# US equities show strong open-to-close momentum (9:30-16:00 ET).
# Crypto has no daily open/close, but we can use UTC session windows.
# Hypothesis: UTC 08:00-16:00 (London/NY overlap) has stronger trend persistence.
# Use 4h timeframe with session filter + price action + volume.
# Only trade breakouts of prior 4h high/low during active session.
# Exit on session end or reversal.
# Expected: fewer whipsaws, better risk/reward.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = pd.DatetimeIndex(prices['open_time'])
    hours = open_time.hour
    
    # Session: UTC 08:00-16:00 (London/NY overlap)
    in_session = (hours >= 8) & (hours < 16)
    
    # Prior 4h high/low (using same timeframe, shifted by 1 bar)
    # For 4h data, prior bar is exactly the previous 4h candle
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Volume filter: above average of last 20 bars
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # 25% position
    
    for i in range(20, n):
        if not in_session[i]:
            # Force flat outside session
            if position != 0:
                position = 0
                signals[i] = 0.0
            continue
        
        # Skip if no prior data
        if np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Long: break above prior 4h high with volume
        if close[i] > prior_high[i] and volume[i] > vol_ma[i] and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short: break below prior 4h low with volume
        elif close[i] < prior_low[i] and volume[i] > vol_ma[i] and position >= 0:
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or session end next bar
        elif position == 1 and (close[i] < prior_low[i] or (i+1 < n and not in_session[i+1])):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > prior_high[i] or (i+1 < n and not in_session[i+1])):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Session_Breakout_Volume"
timeframe = "4h"
leverage = 1.0