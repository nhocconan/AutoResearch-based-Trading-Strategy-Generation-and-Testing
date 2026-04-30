#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 12h trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Long when %R < -80 and rising from oversold.
# Short when %R > -20 and falling from overbought. 12h EMA50 filters for intermediate trend alignment.
# Volume spike ensures institutional participation. Discrete sizing 0.25 balances return and drawdown.
# Target: 60-100 total trades over 4 years (15-25/year). Works in bull via selective longs on dips,
# and in bear via selective shorts on rallies, avoiding chop via trend filter.

name = "6h_WilliamsR_ME_12hEMA50_VolumeSpike_v2"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Williams %R(14) - momentum oscillator
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Calculate 12h EMA(50) for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(14, 20, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Williams %R < -80 (oversold) AND rising from oversold
                if (curr_williams_r < -80 and 
                    curr_williams_r > williams_r[i-1] and  # %R rising (momentum improving)
                    curr_close > curr_ema_50_12h):  # Above 12h EMA50 (bullish bias)
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Williams %R > -20 (overbought) AND falling from overbought
                elif (curr_williams_r > -20 and 
                      curr_williams_r < williams_r[i-1] and  # %R falling (momentum deteriorating)
                      curr_close < curr_ema_50_12h):  # Below 12h EMA50 (bearish bias)
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -50 (leaving oversold zone) OR loses trend
            if (curr_williams_r > -50 or  # Exiting oversold, taking profit
                curr_close < curr_ema_50_12h):  # Lost intermediate uptrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (leaving overbought zone) OR loses trend
            if (curr_williams_r < -50 or  # Exiting overbought, taking profit
                curr_close > curr_ema_50_12h):  # Lost intermediate downtrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals