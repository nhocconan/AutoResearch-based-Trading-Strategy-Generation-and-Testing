#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) on 6h
# - Entry: Long when Bull Power > 0 AND Bear Power rising (less negative) + 1w EMA trend up + volume > 1.5x 20-period average
#          Short when Bear Power < 0 AND Bull Power falling (less positive) + 1w EMA trend down + volume > 1.5x 20-period average
# - Exit: Reverse signal or Elder Power crosses zero
# - Uses weekly EMA for major trend filter to avoid counter-trend trades in both bull/bear markets
# - Volume confirmation ensures breakouts have conviction
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within HARD MAX: 300 total
# - Works in bull markets via long signals in uptrend, bear markets via short signals in downtrend

name = "6h_1w_elderray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w OHLC for EMA
    close_1w = df_1w['close'].values
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_6h - ema_13_6h  # High - EMA
    bear_power = low_6h - ema_13_6h   # Low - EMA
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 6h volume moving average (20-period)
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF data to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_6h)  # Using 1w df for alignment but 6h values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmation = volume_6h[i] > 1.5 * volume_ma_aligned[i]
        
        # Trend filter: 1w EMA direction
        if i >= 1:
            ema_trend_up = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            ema_trend_down = ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power rising (less negative) + uptrend + volume
            bear_power_rising = i >= 1 and bear_power[i] > bear_power[i-1]
            if (bull_power[i] > 0 and 
                bear_power_rising and 
                ema_trend_up and 
                volume_confirmation):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 AND Bull Power falling (less positive) + downtrend + volume
            elif (bear_power[i] < 0 and 
                  i >= 1 and bull_power[i] < bull_power[i-1] and 
                  ema_trend_down and 
                  volume_confirmation):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: reverse Elder Power signal or zero-cross
            if position == 1:  # Long position
                # Exit when Bull Power <= 0 OR Bear Power >= 0 (momentum fading)
                if bull_power[i] <= 0 or bear_power[i] >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                # Exit when Bear Power >= 0 OR Bull Power <= 0 (momentum fading)
                if bear_power[i] >= 0 or bull_power[i] <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals