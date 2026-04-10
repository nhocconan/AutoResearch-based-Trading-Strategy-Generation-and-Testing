#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA trend filter + volume confirmation
# - Williams Alligator: Jaw (EMA13, 8-bar offset), Teeth (EMA8, 5-bar offset), Lips (EMA5, 3-bar offset)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND 1d close > 1d EMA50 AND volume > 1.5x 20-bar avg
# - Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND 1d close < 1d EMA50 AND volume > 1.5x 20-bar avg
# - Exit when Alligator lines cross (Lips crosses Teeth) OR volume drops below 0.7x average
# - Uses 1d trend filter to avoid counter-trend trades in bear markets (2025+)
# - Moderate volume threshold (1.5x) balances signal quality and trade frequency (target: 15-25 trades/year)
# - Williams Alligator catches trends early and avoids whipsaws in ranging markets

name = "12h_1d_williams_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data properly
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Align them to 12h timeframe
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 12h timeframe
    close_s = prices['close']
    # Jaw: EMA13 with 8-bar offset
    jaw = close_s.ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    # Teeth: EMA8 with 5-bar offset
    teeth = close_s.ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    # Lips: EMA5 with 3-bar offset
    lips = close_s.ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long entry: bullish alignment + price > Lips + 1d uptrend + volume spike
            if (bullish_align and 
                prices['close'].iloc[i] > lips[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish alignment + price < Lips + 1d downtrend + volume spike
            elif (bearish_align and 
                  prices['close'].iloc[i] < lips[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Alligator lines cross (Lips crosses Teeth) OR weak volume
            lips_teeth_cross = (lips[i] - teeth[i]) * (lips[max(i-1,0)] - teeth[max(i-1,0)]) <= 0
            
            if position == 1:  # Long position
                if lips_teeth_cross or vol_weak.iloc[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if lips_teeth_cross or vol_weak.iloc[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals