#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d Trend and Volume Spike
# - Bull Power = High - EMA(13), Bear Power = Low - EMA(13) on 1d timeframe
# - Long when Bull Power > 0 and Bear Power < 0 (bullish bias) + price > EMA(13) on 6h + volume spike
# - Short when Bear Power < 0 and Bull Power < 0 (bearish bias) + price < EMA(13) on 6h + volume spike
# - Uses 1d EMA13 to derive power, avoiding look-ahead via proper alignment
# - Volume spike filters for conviction, reducing false breakouts
# - Works in bull/bear by aligning with 1d trend via Elder Ray
# - Target: 20-35 trades/year to stay within 6f limits (80-140 total over 4 years)

name = "6h_ElderRay_Trend_1dEMA13_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA13 (used to calculate Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1d close
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align to 6t timeframe (wait for completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # 6h EMA13 for trend filter on entry timeframe
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # enough for EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_13_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bullish bias) + Bear Power < 0 (confirm) + price > EMA13_6h + volume spike
            long_cond = (bull_power_aligned[i] > 0 and 
                        bear_power_aligned[i] < 0 and
                        close[i] > ema_13_6h[i] and
                        volume_spike[i])
            
            # Short: Bear Power < 0 (bearish bias) + Bull Power < 0 (confirm) + price < EMA13_6h + volume spike
            short_cond = (bear_power_aligned[i] < 0 and 
                         bull_power_aligned[i] < 0 and
                         close[i] < ema_13_6h[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power becomes positive (loss of bullish bias) or price < EMA13_6h
            if bear_power_aligned[i] > 0 or close[i] < ema_13_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power becomes positive (loss of bearish bias) or price > EMA13_6h
            if bull_power_aligned[i] > 0 or close[i] > ema_13_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals