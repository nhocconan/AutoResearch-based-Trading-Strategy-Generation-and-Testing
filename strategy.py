#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1w trend filter + volume confirmation
# - Williams Alligator: Jaw (EMA13, offset8), Teeth (EMA8, offset5), Lips (EMA5, offset3)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND 1w close > 1w EMA34 AND volume > 1.5x 20-bar average
# - Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND 1w close < 1w EMA34 AND volume > 1.5x 20-bar average
# - Exit when Alligator lines cross (Lips crosses Teeth) OR volume drops below 0.8x average
# - Uses 1w trend filter to avoid counter-trend trades and targets 12-30 trades/year (50-120 total over 4 years)
# - Williams Alligator catches trends early with smoothed EMAs, reducing whipsaw in choppy markets

name = "6h_1w_williams_alligator_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.8x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1w data properly
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Align them to 6h timeframe
    h_1w_aligned = align_htf_to_ltf(prices, df_1w, h_1w)
    l_1w_aligned = align_htf_to_ltf(prices, df_1w, l_1w)
    c_1w_aligned = align_htf_to_ltf(prices, df_1w, c_1w)
    
    # Pre-compute 1w EMA(34) for trend filter
    ema34_1w = pd.Series(c_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Williams Alligator parameters
    jaw_period = 13
    jaw_offset = 8
    teeth_period = 8
    teeth_offset = 5
    lips_period = 5
    lips_offset = 3
    
    # Pre-compute Alligator components on close prices
    close_series = prices['close']
    
    # Jaw: EMA(13) offset by 8 bars
    jaw = close_series.ewm(span=jaw_period, adjust=False, min_periods=jaw_period).mean().shift(jaw_offset).values
    
    # Teeth: EMA(8) offset by 5 bars
    teeth = close_series.ewm(span=teeth_period, adjust=False, min_periods=teeth_period).mean().shift(teeth_offset).values
    
    # Lips: EMA(5) offset by 3 bars
    lips = close_series.ewm(span=lips_period, adjust=False, min_periods=lips_period).mean().shift(lips_offset).values
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed 1w bar values for trend filter
        # Since 6h timeframe, there are 28 bars per 1w bar (7 days * 4 bars/day)
        if i >= 28:  # Need at least 28 6h bars to get previous 1w bar's data
            # Get index of previous completed 1w bar
            prev_1w_idx = i - 28  # Look back 28 bars (one 1w period)
            
            if prev_1w_idx >= 0 and not (np.isnan(h_1w_aligned[prev_1w_idx]) or 
                                        np.isnan(l_1w_aligned[prev_1w_idx]) or 
                                        np.isnan(c_1w_aligned[prev_1w_idx])):
                # 1w trend filter: use previous completed 1w close vs EMA
                prev_1w_close = c_1w_aligned[prev_1w_idx]
                prev_1w_ema34 = ema34_1w_aligned[prev_1w_idx]
                
                # Williams Alligator signals
                lips_val = lips[i]
                teeth_val = teeth[i]
                jaw_val = jaw[i]
                
                # Bullish alignment: Lips > Teeth > Jaw
                bullish_alignment = lips_val > teeth_val > jaw_val
                # Bearish alignment: Lips < Teeth < Jaw
                bearish_alignment = lips_val < teeth_val < jaw_val
                
                if position == 0:  # Flat - look for new entries
                    # Long entry: bullish alignment AND price > Lips AND 1w uptrend AND volume spike
                    if (bullish_alignment and 
                        prices['close'].iloc[i] > lips_val and 
                        prev_1w_close > prev_1w_ema34 and 
                        vol_spike.iloc[i]):
                        position = 1
                        signals[i] = 0.25
                    # Short entry: bearish alignment AND price < Lips AND 1w downtrend AND volume spike
                    elif (bearish_alignment and 
                          prices['close'].iloc[i] < lips_val and 
                          prev_1w_close < prev_1w_ema34 and 
                          vol_spike.iloc[i]):
                        position = -1
                        signals[i] = -0.25
                else:  # Have position - look for exit
                    # Exit conditions:
                    # 1. Alligator lines cross (Lips crosses Teeth) - trend weakening
                    # 2. Volume drops below 0.8x average (loss of momentum)
                    lips_cross_teeth = (lips_val > teeth_val and position == -1) or (lips_val < teeth_val and position == 1)
                    
                    if position == 1:  # Long position
                        if lips_cross_teeth or vol_weak.iloc[i]:
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = 0.25  # Hold long
                    elif position == -1:  # Short position
                        if lips_cross_teeth or vol_weak.iloc[i]:
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = -0.25  # Hold short
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Not enough data yet, hold flat
            signals[i] = 0.0
    
    return signals