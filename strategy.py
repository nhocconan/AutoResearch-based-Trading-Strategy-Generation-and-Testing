#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Uses discrete sizing 0.25. Camarilla pivots from 12h provide institutional levels; breakouts at R3/S3
# with volume confirmation capture institutional participation. 12h EMA50 ensures trend alignment.
# Designed for 6h timeframe to balance trade frequency and capture multi-day moves in BTC/ETH.
# Works in bull/bear via trend filter and volume confirmation to avoid false breakouts.

name = "6h_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike_v1"
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
    
    # Get 12h data for Camarilla calculation and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior completed 12h bar
    # Camarilla: based on prior bar's high, low, close
    prior_high = np.roll(df_12h['high'].values, 1)
    prior_low = np.roll(df_12h['low'].values, 1)
    prior_close = np.roll(df_12h['close'].values, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # True range for Camarilla calculation
    tr = prior_high - prior_low
    # Camarilla levels
    camarilla_h5 = prior_close + (tr * 1.1 / 2)  # R4
    camarilla_h4 = prior_close + (tr * 1.1 / 4)  # R3
    camarilla_h3 = prior_close + (tr * 1.1 / 6)  # R2
    camarilla_l3 = prior_close - (tr * 1.1 / 6)  # S2
    camarilla_l4 = prior_close - (tr * 1.1 / 4)  # S3
    camarilla_l5 = prior_close - (tr * 1.1 / 2)  # S4
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    h5_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h5)
    l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l5)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0x 30-bar average (on 6h data)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup
        # Skip if any value is NaN
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            continue
            
        # Get current values
        h4 = h4_aligned[i]  # R3 level
        l4 = l4_aligned[i]  # S3 level
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Entry conditions
        # Long: break above R3 with volume spike and above 12h EMA50
        long_entry = (close[i] > h4) and vol_spike and (close[i] > ema_trend)
        # Short: break below S3 with volume spike and below 12h EMA50
        short_entry = (close[i] < l4) and vol_spike and (close[i] < ema_trend)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below R3 or trend reversal
            if close[i] < h4 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above S3 or trend reversal
            if close[i] > l4 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals