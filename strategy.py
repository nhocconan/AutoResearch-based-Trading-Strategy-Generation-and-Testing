#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot (R1/S1) breakout with 1d trend filter and volume spike
    # Camarilla levels identify key intraday support/resistance
    # 1d EMA34 filters for long-term trend direction (works in bull/bear markets)
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Breakouts from pivot levels with trend alignment and volume provide edge
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 12h bar
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # Shift to get previous bar's typical price
    prev_typical = np.roll(typical_price, 1)
    prev_typical[0] = np.nan  # First value invalid
    
    # Camarilla levels (based on previous bar)
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    camarilla_r1 = prev_close + camarilla_range * 1.1 / 12
    camarilla_s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 + volume spike + price above EMA34 (uptrend)
            if close[i] > camarilla_r1[i] and vol_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 + volume spike + price below EMA34 (downtrend)
            elif close[i] < camarilla_s1[i] and vol_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to previous close (mean reversion) or trend reversal
            if position == 1:
                if close[i] < prev_close[i]:  # Return to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > prev_close[i]:  # Return to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0