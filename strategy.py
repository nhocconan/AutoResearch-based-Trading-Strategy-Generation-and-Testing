#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d trend filter
# - Camarilla levels calculated from prior 12h bar's range (HLC of completed 12h bar)
# - Long when price breaks above R4 with volume > 1.3x 12h average AND 1d close > 1d EMA50
# - Short when price breaks below S4 with volume > 1.3x 12h average AND 1d close < 1d EMA50
# - Exit when price retests the 12h midpoint (PP) or volume drops below average
# - Uses completed 12h bar for pivot calculation to avoid look-ahead
# - Volume confirmation from 12h timeframe to ensure institutional participation
# - 1d EMA50 filter ensures alignment with dominant trend
# - Targets 12-25 trades/year (48-100 total over 4 years) to avoid fee drag
# - Camarilla pivots work well in both trending and ranging markets when combined with volume and trend filters

name = "6h_12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h Camarilla pivot levels from prior completed 12h bar
    # Typical price for pivot: (H + L + C) / 3
    typical_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
    range_12h = df_12h['high'] - df_12h['low']
    
    # Camarilla levels
    camarilla_pp = typical_12h
    camarilla_r4 = camarilla_pp + (range_12h * 1.1 / 2)
    camarilla_s4 = camarilla_pp - (range_12h * 1.1 / 2)
    camarilla_r3 = camarilla_pp + (range_12h * 1.1 / 4)
    camarilla_s3 = camarilla_pp - (range_12h * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (available after 12h bar closes)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pp.values)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4.values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3.values)
    
    # Pre-compute 12h volume confirmation: > 1.3x 20-period average
    volume_20_avg_12h = df_12h['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = df_12h['volume'] > (1.3 * volume_20_avg_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.values)
    
    # Pre-compute 12h volume normal: < average volume for exit
    vol_normal_12h = df_12h['volume'] < volume_20_avg_12h
    vol_normal_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_normal_12h.values)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_spike_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > R4 with volume spike AND daily uptrend
            if (prices['close'].iloc[i] > camarilla_r4_aligned[i] and 
                vol_spike_12h_aligned.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < S4 with volume spike AND daily downtrend
            elif (prices['close'].iloc[i] < camarilla_s4_aligned[i] and 
                  vol_spike_12h_aligned.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retreats to pivot point (mean reversion signal)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < camarilla_pp_aligned[i] or 
                    vol_normal_12h_aligned.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > camarilla_pp_aligned[i] or 
                    vol_normal_12h_aligned.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals