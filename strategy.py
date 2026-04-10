#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray power + volume confirmation
# - Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Ray bull power > 0 AND volume > 1.5x 20-bar avg
# - Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Ray bear power < 0 AND volume > 1.5x 20-bar avg
# - Exit when Alligator alignment breaks (jaws-teeth-lips not monotonic) OR Elder Ray power reverses sign
# - Uses 1w EMA50 for trend filter to avoid counter-trend trades in choppy markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)
# - Williams Alligator identifies trend phases; Elder Ray measures bull/bear power; volume confirms conviction

name = "1d_1w_alligator_elder_ray_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams Alligator (13,8,5 SMAs smoothed)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Pre-compute Elder Ray Power (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(ema50_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Alligator bullish AND Elder Ray bull power > 0 AND volume spike
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and  # Bullish alignment
                bull_power[i] > 0 and 
                vol_spike[i] and
                close[i] > ema50_1w_aligned[i]):  # Above weekly trend
                position = 1
                signals[i] = 0.25
            # Short when Alligator bearish AND Elder Ray bear power < 0 AND volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and  # Bearish alignment
                  bear_power[i] < 0 and 
                  vol_spike[i] and
                  close[i] < ema50_1w_aligned[i]):  # Below weekly trend
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Alligator alignment breaks OR Elder Ray power reverses
            exit_signal = False
            if position == 1:  # Long position
                if not (jaw[i] < teeth[i] and teeth[i] < lips[i]) or bull_power[i] <= 0:
                    exit_signal = True
            elif position == -1:  # Short position
                if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or bear_power[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals