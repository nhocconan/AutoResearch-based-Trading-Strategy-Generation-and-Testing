#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d trend filter and volume confirmation
# Elder Ray measures bullish/bearish power as price relative to EMA(13):
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with bullish 1d EMA(34) trend and volume spike
# Short when Bear Power < 0 and falling, Bull Power < 0 and rising, with bearish 1d EMA(34) trend and volume spike
# Works in bull/bear via 1d trend filter and momentum-based signals targeting 15-35 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Elder Ray components on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d EMA(34) for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (20-period on 6h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Elder Ray momentum (change in power)
    bull_power_change = np.diff(bull_power, prepend=0)
    bear_power_change = np.diff(bear_power, prepend=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power < 0, bullish 1d trend, volume spike
            if (bull_power[i] > 0 and 
                bull_power_change[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling, Bull Power < 0, bearish 1d trend, volume spike
            elif (bear_power[i] < 0 and 
                  bear_power_change[i] < 0 and 
                  bull_power[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Elder Ray signals reverse or trend changes
            if position == 1:
                # Exit on Bull Power <= 0 or Bear Power >= 0 or trend reversal
                if (bull_power[i] <= 0 or 
                    bear_power[i] >= 0 or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Bear Power >= 0 or Bull Power <= 0 or trend reversal
                if (bear_power[i] >= 0 or 
                    bull_power[i] <= 0 or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0