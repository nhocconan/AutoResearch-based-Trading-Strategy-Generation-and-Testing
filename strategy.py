#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume spike
    # Elder Ray measures bull/bear power relative to EMA13
    # EMA50 on 1d filters for long-term trend direction
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in bull/bear: Elder Ray divergence + volume confirms trend strength
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Elder Ray on 6h data (EMA13 for reference)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND rising + Bear Power < 0 AND volume spike + price above EMA50
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and bear_power[i] < 0 and vol_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND falling + Bull Power < 0 AND volume spike + price below EMA50
            elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and bull_power[i] < 0 and vol_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Power signals weaken (trend losing momentum)
            if position == 1:
                if bull_power[i] <= 0 or bull_power[i] < bull_power[i-1]:  # Bull power weakening
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bear_power[i] >= 0 or bear_power[i] > bear_power[i-1]:  # Bear power weakening
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0