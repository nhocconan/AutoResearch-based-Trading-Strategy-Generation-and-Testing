#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike
# - Primary: 6h Williams %R(14) < -80 for longs, > -20 for shorts (oversold/overbought)
# - Trend filter: 1w EMA(34) slope aligned with trade direction (long in uptrend, short in downtrend)
# - Volume confirmation: 6h volume > 1.5x 20-period volume MA to avoid false signals in low volume
# - Exit: Williams %R returns to -50 level or opposite extreme
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Williams %R captures mean reversion in ranging markets, EMA filter ensures
#   we trade with weekly trend, volume confirmation avoids whipsaws

name = "6h_1w_williamsr_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R(14) for 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_34_slope_1w = np.diff(ema_34_1w_aligned, prepend=ema_34_1w_aligned[0])
    
    # Calculate 6h volume confirmation: volume > 1.5x 20-period volume MA
    volume_ma_20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_slope_1w[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.5x 20-period MA
        vol_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        if position == 0:  # Flat - look for new Williams %R extremes
            # Long entry: Williams %R < -80 (oversold) + vol confirmation + weekly uptrend
            if williams_r[i] < -80 and vol_confirm and ema_34_slope_1w[i] > 0:
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + vol confirmation + weekly downtrend
            elif williams_r[i] > -20 and vol_confirm and ema_34_slope_1w[i] < 0:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R returns to -50 level or reaches opposite extreme
            if position == 1:  # Long position
                if williams_r[i] >= -50 or williams_r[i] > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r[i] <= -50 or williams_r[i] < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals