#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w volume confirmation and 1w chop regime filter
# - Primary: 1d price above/below Alligator lines (Jaw, Teeth, Lips) for trend direction
# - Volume filter: 1w volume > 1.3x 50-period volume MA to confirm institutional participation
# - Regime filter: 1w Choppiness Index(34) < 38.2 (trending market) to avoid whipsaw in ranging markets
# - Exit: Price crosses Alligator midline (Teeth line) in opposite direction
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Alligator adapts to volatility, chop filter ensures we only trade trends
# - Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe

name = "1d_1w_alligator_volume_chop_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Williams Alligator (1d timeframe)
    # Jaw: 13-period SMMA smoothed 8 periods ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean().shift(8)
    # Teeth: 8-period SMMA smoothed 5 periods ahead
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean().shift(5)
    # Lips: 5-period SMMA smoothed 3 periods ahead
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean().shift(3)
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    midline = teeth_values  # Use Teeth as midline for exit
    
    # Align Alligator lines to 1d timeframe (already aligned as we calculate on 1d)
    # No need for HTF alignment since we're using primary timeframe data
    
    # Calculate 1w volume confirmation: volume > 1.3x 50-period volume MA
    volume_ma_50_1w = pd.Series(volume_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    volume_ma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_50_1w)
    
    # Calculate 34-period Choppiness Index for regime filter (using 1w data)
    high_low_1w = high_1w - low_1w
    high_close_1w = np.abs(high_1w - np.roll(close_1w, 1))
    low_close_1w = np.abs(low_1w - np.roll(close_1w, 1))
    
    # Handle first element
    high_low_1w[0] = high_1w[0] - low_1w[0]
    high_close_1w[0] = np.abs(high_1w[0] - close_1w[0])
    low_close_1w[0] = np.abs(low_1w[0] - close_1w[0])
    
    tr_1w = np.maximum(high_low_1w, np.maximum(high_close_1w, low_close_1w))
    atr_sum_1w = pd.Series(tr_1w).rolling(window=34, min_periods=34).sum().values
    max_high_1w = pd.Series(high_1w).rolling(window=34, min_periods=34).max().values
    min_low_1w = pd.Series(low_1w).rolling(window=34, min_periods=34).min().values
    
    # Avoid division by zero
    range_hl_1w = max_high_1w - min_low_1w
    range_hl_1w = np.where(range_hl_1w == 0, 1e-10, range_hl_1w)
    
    chop_1w = 100 * np.log10(atr_sum_1w / range_hl_1w) / np.log10(34)
    chop_filter = chop_1w < 38.2  # Chop < 38.2 indicates trending market
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or np.isnan(lips_values[i]) or
            np.isnan(volume_ma_50_1w_aligned[i]) or np.isnan(chop_1w[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1w volume > 1.3x 50-period MA
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)
        vol_confirm = vol_1w_current[i] > 1.3 * volume_ma_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price above all Alligator lines (Lips > Teeth > Jaw) + vol confirm + chop filter
            if lips_values[i] > teeth_values[i] and teeth_values[i] > jaw_values[i] and vol_confirm and chop_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price below all Alligator lines (Lips < Teeth < Jaw) + vol confirm + chop filter
            elif lips_values[i] < teeth_values[i] and teeth_values[i] < jaw_values[i] and vol_confirm and chop_filter[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at midline cross
            # Exit: price crosses midline (Teeth) in opposite direction
            if position == 1:  # Long position
                if close[i] <= midline[i]:  # Price crosses below Teeth
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= midline[i]:  # Price crosses above Teeth
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals