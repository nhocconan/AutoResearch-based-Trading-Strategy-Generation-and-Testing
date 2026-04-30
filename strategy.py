#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Uses actual Camarilla pivot calculation from prior day's range
# Long when price breaks above R3 with volume spike and price > 1d EMA34
# Short when price breaks below S3 with volume spike and price < 1d EMA34
# Discrete sizing 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    # Need prior day's high, low, close (not current forming bar)
    df_1d_shift = df_1d.shift(1)  # Shift to get prior completed day
    prior_high = df_1d_shift['high'].values
    prior_low = df_1d_shift['low'].values
    prior_close = df_1d_shift['close'].values
    
    # True range for Camarilla calculation
    tr = np.maximum(prior_high - prior_low,
                    np.maximum(np.abs(prior_high - prior_close),
                               np.abs(prior_low - prior_close)))
    
    # Camarilla levels
    R3 = prior_close + (tr * 1.1 / 4)
    S3 = prior_close - (tr * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (wait for prior day to complete)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3, additional_delay_bars=1)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3, additional_delay_bars=1)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34 = ema_34_1d_aligned[i]
        curr_R3 = R3_aligned[i]
        curr_S3 = S3_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above R3 AND price > 1d EMA34 (uptrend)
                if curr_high > curr_R3 and curr_close > curr_ema_34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below S3 AND price < 1d EMA34 (downtrend)
                elif curr_low < curr_S3 and curr_close < curr_ema_34:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below 1d EMA34 (trend change) or touches S3 (mean reversion)
            if curr_close < curr_ema_34 or curr_low <= curr_S3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above 1d EMA34 (trend change) or touches R3 (mean reversion)
            if curr_close > curr_ema_34 or curr_high >= curr_R3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals