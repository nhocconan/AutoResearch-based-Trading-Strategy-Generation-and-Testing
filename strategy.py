#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla R3/S3 Breakout with Daily Volume Spike
# Weekly Camarilla levels provide significant support/resistance that price respects
# Breakouts above R3 or below S3 with volume confirmation indicate strong momentum
# Daily volume spike (>2x 20-period average) validates institutional participation
# Works in bull/bear markets as breakouts capture volatility expansion
# Target: 12-25 trades/year (48-100 total over 4 years)

name = "6h_WeeklyCamarilla_R3S3_Breakout_DailyVolSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly and daily calculations
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on previous week OHLC)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    weekly_range = high_1w - low_1w
    camarilla_r3 = close_1w + 1.1 * weekly_range * 1.1 / 4
    camarilla_s3 = close_1w - 1.1 * weekly_range * 1.1 / 4
    
    # Align weekly Camarilla levels to 6h timeframe (wait for weekly bar close)
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Calculate daily volume spike: volume > 2x 20-period average
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (2.0 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_spike_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_vol_spike = vol_spike_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish breakout: price breaks above weekly R3 with volume spike
            if curr_close > curr_r3 and curr_vol_spike:
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price breaks below weekly S3 with volume spike
            elif curr_close < curr_s3 and curr_vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit when price returns to weekly midline
            # Exit when price crosses below weekly R3 (breakout failed) or reaches midpoint
            weekly_mid = (r3_aligned[i] + s3_aligned[i]) / 2
            if curr_close < weekly_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price returns to weekly midline
            # Exit when price crosses above weekly S3 (breakout failed) or reaches midpoint
            weekly_mid = (r3_aligned[i] + s3_aligned[i]) / 2
            if curr_close > weekly_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals