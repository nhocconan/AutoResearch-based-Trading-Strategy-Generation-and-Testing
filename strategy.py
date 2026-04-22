#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike
    # Camarilla levels provide high-probability reversal zones in both trending and ranging markets
    # EMA34 on 1d filters for long-term trend direction
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Target: 20-40 trades/year with high win rate
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3
    # Use previous day's typical price for today's levels
    typical_price_shifted = np.roll(typical_price, 1)
    typical_price_shifted[0] = typical_price[0]  # first bar
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # We focus on R1 and S1 for breakout entries
    R1 = typical_price_shifted + 1.1 * (high - low) / 12
    S1 = typical_price_shifted - 1.1 * (high - low) / 12
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(R1[i]) or 
            np.isnan(S1[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 + volume spike + price above EMA34 (uptrend)
            if close[i] > R1[i] and vol_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 + volume spike + price below EMA34 (downtrend)
            elif close[i] < S1[i] and vol_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to previous day's typical price (pivot point) or trend reversal
            if position == 1:
                if close[i] < typical_price_shifted[i]:  # Return to pivot
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > typical_price_shifted[i]:  # Return to pivot
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0