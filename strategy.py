#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1w EMA50 trend filter and volume spike confirmation.
# Jaw = EMA(13,8), Teeth = EMA(8,5), Lips = EMA(5,3) of median price.
# Long when Lips > Teeth > Jaw (bullish alignment) AND 1w close > EMA50 (bullish trend) AND volume > 2.0x 20-bar average.
# Short when Lips < Teeth < Jaw (bearish alignment) AND 1w close < EMA50 (bearish trend) AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Alligator identifies trend via SMAs with forward shift; EMA50 filters higher-timeframe trend; volume spike confirms momentum.
# Primary timeframe: 12h, HTF: 1w for EMA trend filter.

name = "12h_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator (12h)
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).ewm(span=13, adjust=False).mean().values
    jaw = np.roll(jaw, 8)  # shift forward by 8 bars
    teeth = pd.Series(median_price).ewm(span=8, adjust=False).mean().values
    teeth = np.roll(teeth, 5)  # shift forward by 5 bars
    lips = pd.Series(median_price).ewm(span=5, adjust=False).mean().values
    lips = np.roll(lips, 3)  # shift forward by 3 bars
    
    # 1w EMA50 trend filter
    prev_close = df_1w['close'].values
    ema_50 = pd.Series(prev_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: current 12h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and indicators
    
    for i in range(start_idx, n):
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = curr_close > ema_50_aligned[i]
        bearish_trend = curr_close < ema_50_aligned[i]
        
        # Alligator alignment
        bullish_alignment = (curr_lips > curr_teeth) and (curr_teeth > curr_jaw)
        bearish_alignment = (curr_lips < curr_teeth) and (curr_teeth < curr_jaw)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Bullish alignment AND bullish trend AND volume confirmation
            if (bullish_alignment and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND bearish trend AND volume confirmation
            elif (bearish_alignment and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish alignment OR trend turns bearish
            if (bearish_alignment or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish alignment OR trend turns bullish
            if (bullish_alignment or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals