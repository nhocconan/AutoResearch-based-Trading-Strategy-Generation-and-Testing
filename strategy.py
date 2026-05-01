#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND 1w close > EMA50 AND volume > 2.0x 20-bar average.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND 1w close < EMA50 AND volume > 2.0x 20-bar average.
# Williams Alligator uses SMAs of median price: jaws=SMA13, teeth=SMA8, lips=SMA5.
# Discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Volume spike threshold set to 2.0x to reduce false signals and improve quality.
# Works in bull markets (trend continuation via Alligator alignment) and bear markets (mean reversion at extremes via price/lips).
# Primary timeframe: 12h, HTF: 1w for trend filter.

name = "12h_Williams_Alligator_1wEMA50_Trend_VolumeSpike_v1"
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
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator components (SMA of median price)
    # Jaws: SMA13, Teeth: SMA8, Lips: SMA5
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 1w EMA50 calculation
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 1w close aligned for trend bias
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Volume confirmation: current 12h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Alligator and indicators
    
    for i in range(start_idx, n):
        if np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema_aligned[i]) or np.isnan(close_1w_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Williams Alligator alignment signals
        bullish_alignment = (jaws[i] < teeth[i]) and (teeth[i] < lips[i])  # jaws < teeth < lips
        bearish_alignment = (jaws[i] > teeth[i]) and (teeth[i] > lips[i])  # jaws > teeth > lips
        
        # Trend filter: use 1w close vs its EMA50 for bias
        bullish_bias = close_1w_aligned[i] > ema_aligned[i]  # 1w close above its EMA50 = bullish
        bearish_bias = close_1w_aligned[i] < ema_aligned[i]  # 1w close below its EMA50 = bearish
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator alignment AND price > lips AND bullish bias AND volume confirmation
            if (bullish_alignment and 
                curr_close > lips[i] and 
                bullish_bias and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND price < lips AND bearish bias AND volume confirmation
            elif (bearish_alignment and 
                  curr_close < lips[i] and 
                  bearish_bias and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bearish Alligator alignment OR price < lips (stoploss) OR bearish bias (trend change)
            if (bearish_alignment or 
                curr_close < lips[i] or 
                bearish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish Alligator alignment OR price > lips (stoploss) OR bullish bias (trend change)
            if (bullish_alignment or 
                curr_close > lips[i] or 
                bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals