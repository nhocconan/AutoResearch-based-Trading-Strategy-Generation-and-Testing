#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1w EMA34 trend filter and volume spike confirmation
# Williams %R: measures overbought/oversold levels (-100 to 0)
# Long: %R crosses above -80 from below (oversold bounce) in uptrend (price > 1w EMA34)
# Short: %R crosses below -20 from above (overbought rejection) in downtrend (price < 1w EMA34)
# Volume spike (2.0x 20-period average) confirms momentum
# Discrete sizing 0.25 minimizes fee churn. Works in bull via %R longs with uptrend,
# in bear via %R shorts with downtrend. Target: 12-37 trades/year (50-150 total over 4 years).

name = "12h_WilliamsR_1wEMA34_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 14, 34)  # warmup for volume MA, Williams %R, and 1w EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(williams_r[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_williams_r = williams_r[i]
        curr_ema_34 = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Williams %R crosses above -80 from below (oversold bounce) AND price > 1w EMA34 (uptrend)
                if prev_williams_r <= -80 and curr_williams_r > -80 and curr_close > curr_ema_34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Williams %R crosses below -20 from above (overbought rejection) AND price < 1w EMA34 (downtrend)
                elif prev_williams_r >= -20 and curr_williams_r < -20 and curr_close < curr_ema_34:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Williams %R rises above -20 (overbought) OR price drops below EMA34
            if curr_williams_r >= -20 or curr_close < curr_ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -80 (oversold) OR price rises above EMA34
            if curr_williams_r <= -80 or curr_close > curr_ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals