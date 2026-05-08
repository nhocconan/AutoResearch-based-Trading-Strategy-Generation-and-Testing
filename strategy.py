#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for Camarilla pivot levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily pivot point (PP)
    pp_1d = (high_1d + low_1d + close_1d) / 3
    
    # Calculate Camarilla levels for S1, S2, R1, R2
    # R4 = C + ((H-L)*1.500), R3 = C + ((H-L)*1.250), R2 = C + ((H-L)*1.166), R1 = C + ((H-L)*1.083)
    # S1 = C - ((H-L)*1.083), S2 = C - ((H-L)*1.166), S3 = C - ((H-L)*1.250), S4 = C - ((H-L)*1.500)
    r1_1d = pp_1d + ((high_1d - low_1d) * 1.083)
    r2_1d = pp_1d + ((high_1d - low_1d) * 1.166)
    s1_1d = pp_1d - ((high_1d - low_1d) * 1.083)
    s2_1d = pp_1d - ((high_1d - low_1d) * 1.166)
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume spike indicator (volume > 1.5x 20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values if 'volume_1d' in df_1d.columns else np.zeros_like(close_1d)
    # Since we don't have volume in df_1d from get_htf_data, we'll use price range as proxy for volatility
    price_range_1d = high_1d - low_1d
    range_ma_1d = pd.Series(price_range_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = price_range_1d > (range_ma_1d * 1.5)  # Using price range as volatility proxy
    
    # Align all daily indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Price breaks above R1 with volume spike AND price > EMA34 (uptrend)
            long_cond = (close[i] > r1_aligned[i]) and (vol_spike_aligned[i] > 0.5) and (close[i] > ema34_aligned[i])
            
            # Short entry: Price breaks below S1 with volume spike AND price < EMA34 (downtrend)
            short_cond = (close[i] < s1_aligned[i]) and (vol_spike_aligned[i] > 0.5) and (close[i] < ema34_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below S1 (reversal to support) OR loses volume momentum
            if (close[i] < s1_aligned[i]) or (vol_spike_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above R1 (reversal to resistance) OR loses volume momentum
            if (close[i] > r1_aligned[i]) or (vol_spike_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 levels act as intraday support/resistance magnets. 
# Breakouts with volume expansion indicate institutional participation. 
# EMA34 filter ensures we only trade in direction of daily trend to avoid counter-trend whipsaws.
# Works in bull markets (buying R1 breakouts in uptrend) and bear markets (selling S1 breakdowns in downtrend).
# Volume spike filter reduces false breakouts. 
# Target: 20-50 trades/year to minimize fee decay while capturing meaningful moves.