#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1w trend filter + volume confirmation
# - Williams Alligator: Jaw (EMA13, 8), Teeth (EMA8, 5), Lips (EMA5, 3)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1w close > 1w EMA50 AND volume > 1.5x average
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1w close < 1w EMA50 AND volume > 1.5x average
# - Exit when Alligator lines intertwine (Lips crosses Teeth or Jaw) OR volume drops below 0.7x average
# - Uses 1w trend filter to avoid counter-trend trades in any market regime
# - Higher volume threshold (1.5x) reduces false signals and targets 15-25 trades/year (60-100 total over 4 years)
# - Tight entry conditions to avoid fee drag while maintaining edge in both bull and bear regimes

name = "12h_williams_alligator_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute Williams Alligator components
    close = prices['close'].values
    # Jaw: 13-period EMA, smoothed by 8 periods
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw).ewm(span=8, adjust=False, min_periods=8).mean().values
    # Teeth: 8-period EMA, smoothed by 5 periods
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth).ewm(span=5, adjust=False, min_periods=5).mean().values
    # Lips: 5-period EMA, smoothed by 3 periods
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Williams Alligator conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        intertwining = (lips[i] <= teeth[i] and lips[i] >= jaw[i]) or \
                       (teeth[i] <= lips[i] and teeth[i] >= jaw[i])
        
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish Alligator alignment + 1w uptrend + volume spike
            if (bullish_alignment and 
                prices['close'].iloc[i] > ema50_1w_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish Alligator alignment + 1w downtrend + volume spike
            elif (bearish_alignment and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator lines intertwine (trend weakening)
            # 2. Volume drops below 0.7x average (loss of momentum)
            if intertwining or vol_weak.iloc[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1w trend filter + volume confirmation
# - Williams Alligator: Jaw (EMA13, 8), Teeth (EMA8, 5), Lips (EMA5, 3)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1w close > 1w EMA50 AND volume > 1.5x average
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1w close < 1w EMA50 AND volume > 1.5x average
# - Exit when Alligator lines intertwine (Lips crosses Teeth or Jaw) OR volume drops below 0.7x average
# - Uses 1w trend filter to avoid counter-trend trades in any market regime
# - Higher volume threshold (1.5x) reduces false signals and targets 15-25 trades/year (60-100 total over 4 years)
# - Tight entry conditions to avoid fee drag while maintaining edge in both bull and bear regimes

name = "12h_williams_alligator_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute Williams Alligator components
    close = prices['close'].values
    # Jaw: 13-period EMA, smoothed by 8 periods
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw).ewm(span=8, adjust=False, min_periods=8).mean().values
    # Teeth: 8-period EMA, smoothed by 5 periods
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth).ewm(span=5, adjust=False, min_periods=5).mean().values
    # Lips: 5-period EMA, smoothed by 3 periods
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Williams Alligator conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        intertwining = (lips[i] <= teeth[i] and lips[i] >= jaw[i]) or \
                       (teeth[i] <= lips[i] and teeth[i] >= jaw[i])
        
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish Alligator alignment + 1w uptrend + volume spike
            if (bullish_alignment and 
                prices['close'].iloc[i] > ema50_1w_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish Alligator alignment + 1w downtrend + volume spike
            elif (bearish_alignment and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator lines intertwine (trend weakening)
            # 2. Volume drops below 0.7x average (loss of momentum)
            if intertwining or vol_weak.iloc[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals