#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla Pivot Breakout with Daily Volume Spike
# Weekly Camarilla levels (R3/S3, R4/S4) act as strong support/resistance
# Breakout above R4 or below S4 with daily volume spike indicates institutional participation
# Fade at R3/S3 in ranging markets, breakout continuation at R4/S4 in trending markets
# Works in all regimes: captures both mean reversion at strong levels and breakout momentum
# Target: 12-25 trades/year (48-100 total over 4 years)

name = "6h_WeeklyCamarilla_R3S4_Breakout_DailyVolSpike_v1"
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
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    # Based on previous week's high, low, close
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla multiplier
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    weekly_range = weekly_high - weekly_low
    camarilla_r4 = weekly_close + (weekly_range * 1.1 / 2)
    camarilla_r3 = weekly_close + (weekly_range * 1.1 / 4)
    camarilla_s3 = weekly_close - (weekly_range * 1.1 / 4)
    camarilla_s4 = weekly_close - (weekly_range * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Daily volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(10, 20)  # warmup for weekly and volume indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        
        # Get weekly Camarilla levels
        r4 = r4_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Breakout continuation at R4/S4 with volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above R4
                if curr_close > r4 and curr_high > r4:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below S4
                elif curr_close < s4 and curr_low < s4:
                    signals[i] = -0.25
                    position = -1
            # Mean reversion fade at R3/S3 (no volume spike required)
            else:
                # Fade at R3: price rejects at R3 and starts declining
                if curr_close < r3 and curr_high > r3:
                    signals[i] = -0.25
                    position = -1
                # Fade at S3: price rejects at S3 and starts rising
                elif curr_close > s3 and curr_low < s3:
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price crosses below R3 (mean reversion) or S4 (stop)
            if curr_close < r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price crosses above S3 (mean reversion) or R4 (stop)
            if curr_close > s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals