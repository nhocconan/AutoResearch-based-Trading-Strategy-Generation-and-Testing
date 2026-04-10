#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR trailing stop
# - Long when price breaks above 20-period Donchian high AND volume > 1.3x 20-period volume SMA
# - Short when price breaks below 20-period Donchian low AND volume > 1.3x 20-period volume SMA
# - ATR(14) trailing stop: 2.0x ATR for longs, 2.0x ATR for shorts
# - Session filter: 08-20 UTC to avoid low-volume Asian session
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Donchian channels provide clear structure that works in both bull and bear markets via breakouts

name = "4h_donchian_volume_atr_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above upper band
        breakout_down = close[i] < donchian_low[i-1]  # Break below lower band
        
        if position == 0:  # Flat - look for entry
            # Long: price breaks above upper band AND volume confirmation
            if breakout_up and vol_confirm:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: price breaks below lower band AND volume confirmation
            elif breakout_down and vol_confirm:
                position = -1
                signals[i] = -0.25
                lowest_since_entry[i] = low[i]  # Initialize trailing stop
            else:
                signals[i] = 0.0
                # Carry forward NaN values for tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:  # Long position - look for exit
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            
            # ATR trailing stop: exit if price drops 2.0*ATR below highest high since entry
            trailing_stop = highest_since_entry[i] - 2.0 * atr[i]
            
            if close[i] < trailing_stop:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i]
                lowest_since_entry[i] = lowest_since_entry[i-1]
        else:  # position == -1 (Short position) - look for exit
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            
            # ATR trailing stop: exit if price rises 2.0*ATR above lowest low since entry
            trailing_stop = lowest_since_entry[i] + 2.0 * atr[i]
            
            if close[i] > trailing_stop:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i]
    
    return signals