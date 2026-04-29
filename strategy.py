#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Mean Reversion with 1w EMA Trend Filter and Volume Spike Confirmation
# Long when: Williams %R < -80 (oversold) AND price > 1w EMA34 (uptrend) AND volume > 1.5 * 20-period average volume
# Short when: Williams %R > -20 (overbought) AND price < 1w EMA34 (downtrend) AND volume > 1.5 * 20-period average volume
# Uses Williams %R for mean reversion extremes, 1w EMA for trend alignment, volume spike for confirmation.
# Works in bull/bear via trend filter (only trade in direction of 1w EMA) + mean reversion at extremes.
# Timeframe: 12h (primary), HTF: 1w for EMA calculation.

name = "12h_WilliamsR_MeanReversion_1wEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R on 12h data (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate volume spike: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if 1w EMA data not available
        if np.isnan(ema_34_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit when Williams %R rises above -50 (exiting oversold) OR volume spike ends
            if (curr_williams_r > -50) or (not curr_volume_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (exiting overbought) OR volume spike ends
            if (curr_williams_r < -50) or (not curr_volume_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) AND price > 1w EMA34 (uptrend) AND volume spike
            if (curr_williams_r < -80) and (curr_close > curr_ema_34_1w) and curr_volume_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought) AND price < 1w EMA34 (downtrend) AND volume spike
            elif (curr_williams_r > -20) and (curr_close < curr_ema_34_1w) and curr_volume_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals