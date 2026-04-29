#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold recovery) in bullish 1d regime (price > EMA50) with volume spike
# Short when Williams %R crosses below -20 (overbought decline) in bearish 1d regime (price < EMA50) with volume spike
# Williams %R identifies mean reversion extremes, 1d EMA50 filters for directional bias, volume confirms participation
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag

name = "6h_WilliamsR_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to 6h timeframe (completed 1d bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) on 6h
    williams_window = 14
    highest_high = pd.Series(high).rolling(window=williams_window, min_periods=williams_window).max().values
    lowest_low = pd.Series(low).rolling(window=williams_window, min_periods=williams_window).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * (highest_high - close) / (highest_high - lowest_low), -50)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams = williams_r[i]
        curr_ema_50 = ema_50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend filter: price above/below 1d EMA50
        is_bullish = curr_close > curr_ema_50
        is_bearish = curr_close < curr_ema_50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: Williams %R crosses above -80 from oversold
                if curr_williams > -80 and williams_r[i-1] <= -80 and is_bullish:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -20 from overbought
                elif curr_williams < -20 and williams_r[i-1] >= -20 and is_bearish:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Williams %R crosses below -50 (momentum loss) or crosses above -10 (overbought)
            if curr_williams < -50 and williams_r[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            elif curr_williams > -10:  # Overbought exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Williams %R crosses above -50 (momentum loss) or crosses below -90 (oversold)
            if curr_williams > -50 and williams_r[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            elif curr_williams < -90:  # Oversold exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals