#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot levels + 1d EMA50 trend filter + volume confirmation
    # Camarilla pivots identify key support/resistance levels from prior day
    # EMA50 on 1d filters for intermediate trend direction
    # Volume spike (1.5x 20-period MA) confirms breakout validity
    # Works in both bull and bear: pivot levels act as magnets in ranges, breakouts in trends
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical Camarilla uses previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    hl_range = prev_high - prev_low
    camarilla_r1 = prev_close + hl_range * 1.1 / 12
    camarilla_s1 = prev_close - hl_range * 1.1 / 12
    camarilla_r2 = prev_close + hl_range * 1.1 / 6
    camarilla_s2 = prev_close - hl_range * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume spike + price above EMA50 (uptrend bias)
            if close[i] > r1_aligned[i] and vol_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume spike + price below EMA50 (downtrend bias)
            elif close[i] < s1_aligned[i] and vol_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite S/R level or trend reversal
            if position == 1:
                if close[i] < s1_aligned[i]:  # Break below S1 suggests reversal
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_aligned[i]:  # Break above R1 suggests reversal
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0