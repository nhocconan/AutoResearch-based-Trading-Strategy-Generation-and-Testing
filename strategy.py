#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w volume confirmation and chop regime filter
# - Primary: 1d Williams Alligator (JAWS=EMA13, TEETH=EMA8, LIPS=EMA5) - long when LIPS > TEETH > JAWS, short when LIPS < TEETH < JAWS
# - Volume filter: 1w volume > 1.5x 20-period volume MA to confirm institutional interest
# - Regime filter: Choppiness Index(14) < 38.2 (trending market) for trend following to work
# - Exit: Alligator lines cross (trend exhaustion)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Alligator shows trend direction, chop filter ensures trending environment
# - Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe

name = "1d_1w_alligator_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    volume_1w = df_1w['volume'].values
    
    # Calculate Williams Alligator: JAWS=EMA13, TEETH=EMA8, LIPS=EMA5 (all on Median Price)
    median_price = (high + low) / 2
    jaws = pd.Series(median_price).ewm(span=13, min_periods=13, adjust=False).mean().values  # Blue line
    teeth = pd.Series(median_price).ewm(span=8, min_periods=8, adjust=False).mean().values    # Red line
    lips = pd.Series(median_price).ewm(span=5, min_periods=5, adjust=False).mean().values    # Green line
    
    # Calculate 1w volume confirmation: volume > 1.5x 20-period volume MA
    volume_ma_20_1w = pd.Series(volume_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    # Calculate 14-period Choppiness Index for regime filter (using 1d data)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    
    # Handle first element
    high_low[0] = high[0] - low[0]
    high_close[0] = np.abs(high[0] - close[0])
    low_close[0] = np.abs(low[0] - close[0])
    
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = max_high - min_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    chop_filter = chop < 38.2  # Chop < 38.2 indicates trending market (good for trend following)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaws[i]) or 
            np.isnan(volume_ma_20_1w_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Align 1w volume data for current bar
        volume_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)
        vol_confirm = volume_1w_current[i] > 1.5 * volume_ma_20_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish (Lips > Teeth > Jaws) + vol confirmation + chop filter
            if (lips[i] > teeth[i] and teeth[i] > jaws[i] and 
                vol_confirm and chop_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Alligator bearish (Lips < Teeth < Jaws) + vol confirmation + chop filter
            elif (lips[i] < teeth[i] and teeth[i] < jaws[i] and 
                  vol_confirm and chop_filter[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when Alligator lines cross (trend exhaustion)
            # Exit: Alligator lines cross (Lips crosses Teeth or Teeth crosses Jaws)
            if position == 1:  # Long position
                if lips[i] <= teeth[i] or teeth[i] <= jaws[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if lips[i] >= teeth[i] or teeth[i] >= jaws[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals