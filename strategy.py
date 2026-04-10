#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator: Jaw (EMA13, 8-bar shift), Teeth (EMA8, 5-bar shift), Lips (EMA5, 3-bar shift)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND volume > 1.5x 20-bar average AND 1d close > 1d EMA50
# - Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND volume > 1.5x 20-bar average AND 1d close < 1d EMA50
# - Exit when Alligator alignment breaks OR volume drops below 0.7x average
# - Uses 1d trend filter to avoid counter-trend trades in bear markets (2025+)
# - Williams Alligator is effective in ranging and trending markets, providing clear entry/exit signals
# - Moderate volume threshold (1.5x) balances signal quality and trade frequency (target: 15-25 trades/year)
# - Focus on BTC/ETH; SOL-only strategies are low value and will be discarded

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
    
    # Pre-compute Williams Alligator components on close prices
    close_series = prices['close']
    
    # Jaw: EMA(13) shifted by 8 bars
    jaw = close_series.ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    
    # Teeth: EMA(8) shifted by 5 bars
    teeth = close_series.ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    
    # Lips: EMA(5) shifted by 3 bars
    lips = close_series.ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(h_1d_aligned[i]) or np.isnan(l_1d_aligned[i]) or np.isnan(c_1d_aligned[i])):
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
            
            # Long entry: bullish alignment AND price > Lips AND volume spike AND 1d uptrend
            if (bullish_align and 
                prices['close'].iloc[i] > lips[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish alignment AND price < Lips AND volume spike AND 1d downtrend
            elif (bearish_align and 
                  prices['close'].iloc[i] < lips[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Stay flat
                
        elif position == 1:  # Long position - look for exit
            # Exit conditions:
            # 1. Alligator alignment breaks (not bullish)
            # 2. Price drops below Lips
            # 3. Volume drops below 0.7x average (loss of momentum)
            bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
            if (not bullish_align or 
                prices['close'].iloc[i] < lips[i] or 
                vol_weak.iloc[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Hold long
                
        elif position == -1:  # Short position - look for exit
            # Exit conditions:
            # 1. Alligator alignment breaks (not bearish)
            # 2. Price rises above Lips
            # 3. Volume drops below 0.7x average (loss of momentum)
            bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
            if (not bearish_align or 
                prices['close'].iloc[i] > lips[i] or 
                vol_weak.iloc[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Hold short
    
    return signals