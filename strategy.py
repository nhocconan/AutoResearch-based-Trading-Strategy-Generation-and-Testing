#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA50 trend filter and volume spike
    # Elder Ray measures bull/bear power relative to EMA13 (Bull Power = High - EMA13, Bear Power = EMA13 - Low)
    # EMA50 on 1d filters for medium-term trend direction
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in bull/bear: uses power imbalance with trend and volume confirmation
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull power: high minus EMA13
    bear_power = ema13 - low   # Bear power: EMA13 minus low
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power > 0 (strong bulls) with volume spike and price above 1d EMA50 (uptrend)
            if bull_power[i] > 0 and vol_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear power > 0 (strong bears) with volume spike and price below 1d EMA50 (downtrend)
            elif bear_power[i] > 0 and vol_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Power imbalance reverses
            if position == 1:
                if bull_power[i] <= 0:  # Bulls lose control
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bear_power[i] <= 0:  # Bears lose control
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0