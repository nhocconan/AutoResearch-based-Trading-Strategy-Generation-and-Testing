#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d trend filter and volume confirmation.
# Long when Williams %R(14) crosses above -80 from below, price > 1d EMA50, and volume > 1.5x 20-bar average.
# Short when Williams %R(14) crosses below -20 from above, price < 1d EMA50, and volume > 1.5x 20-bar average.
# Exit on opposite Williams %R extreme (-20 for long, -80 for short) or ATR(14) trailing stop (2.5x ATR).
# Uses 1d EMA50 for trend alignment and 6h ATR for dynamic risk control.
# Target: 12-30 trades/year by requiring confluence of momentum extreme, trend, and volume spike.
# Williams %R identifies overextended conditions; EMA50 filters for trend direction; volume confirms conviction.

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeSpike_ATRStop_v1"
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
    
    # Pre-compute session hours for efficiency (optional, can be removed if not needed)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d data
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for 6h timeframe trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Williams %R(14) for 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close) / (highest_high - lowest_low),
                          -50)  # fallback to midpoint when range is zero
    
    # Calculate volume spike condition: volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (can be adjusted or removed)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_williams_r = williams_r[i]
        curr_ema = ema_50_aligned[i]
        curr_atr = atr[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from below, price above 1d EMA50, volume spike
            if (i > start_idx and 
                williams_r[i-1] <= -80 and 
                curr_williams_r > -80 and 
                curr_close > curr_ema and 
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                long_stop = curr_low - 2.5 * curr_atr  # initial stop below entry
            # Short: Williams %R crosses below -20 from above, price below 1d EMA50, volume spike
            elif (i > start_idx and 
                  williams_r[i-1] >= -20 and 
                  curr_williams_r < -20 and 
                  curr_close < curr_ema and 
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                short_stop = curr_high + 2.5 * curr_atr  # initial stop above entry
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update trailing stop: move stop up to highest high minus 2.5*ATR
            if i == start_idx:
                long_stop = curr_low - 2.5 * curr_atr
            else:
                long_stop = max(long_stop, curr_high - 2.5 * curr_atr)
            # Exit conditions: Williams %R rises above -20 OR stoploss hit
            if (curr_williams_r > -20 or 
                curr_close < long_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update trailing stop: move stop down to lowest low plus 2.5*ATR
            if i == start_idx:
                short_stop = curr_high + 2.5 * curr_atr
            else:
                short_stop = min(short_stop, curr_low + 2.5 * curr_atr)
            # Exit conditions: Williams %R falls below -80 OR stoploss hit
            if (curr_williams_r < -80 or 
                curr_close > short_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals