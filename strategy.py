#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Uses proven Camarilla pivot levels (R3/S3) from daily timeframe for structure,
# 1d EMA34 for higher timeframe trend filter, and volume confirmation for breakout strength.
# Tight entry conditions target 20-40 trades/year to minimize fee drag.
# Works in bull/bear markets via trend filter - only takes breakouts in direction of 1d EMA34.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from 1d data
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But we use prior day's OHLC to avoid look-ahead
    prior_high = df_1d['high'].shift(1).values  # Prior day's high
    prior_low = df_1d['low'].shift(1).values    # Prior day's low
    prior_close = df_1d['close'].shift(1).values # Prior day's close
    
    # Calculate Camarilla levels using prior day's data
    camarilla_range = prior_high - prior_low
    r3 = prior_close + 1.1 * camarilla_range  # Resistance 3
    s3 = prior_close - 1.1 * camarilla_range  # Support 3
    
    # Align Camarilla levels to 4h timeframe (they change only at daily boundaries)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 34, 20)  # warmup for EMA34, Camarilla, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price crosses below 1d EMA34 (trend change)
            # 2. Price re-enters Camarilla levels (breakout failed)
            if (curr_close < curr_ema_34_1d or
                curr_close < curr_r3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price crosses above 1d EMA34 (trend change)
            # 2. Price re-enters Camarilla levels (breakout failed)
            if (curr_close > curr_ema_34_1d or
                curr_close > curr_s3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 + above 1d EMA34 + volume confirm
            if (curr_close > curr_r3 and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Camarilla S3 + below 1d EMA34 + volume confirm
            elif (curr_close < curr_s3 and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals