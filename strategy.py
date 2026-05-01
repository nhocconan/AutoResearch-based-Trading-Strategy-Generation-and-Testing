#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume spike confirmation
# Elder Ray measures bull/bear strength relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND price > 1d EMA34 (uptrend) AND volume > 1.5x 20-period EMA
# Short when Bear Power < 0 AND price < 1d EMA34 (downtrend) AND volume > 1.5x 20-period EMA
# Works in bull/bear: follows the trend via 1d EMA34, volume confirms participation, Elder Ray ensures momentum
# Target trade frequency: ~15-25 trades/year per symbol (60-100 total over 4 years) with 0.25 sizing

name = "6h_ElderRay_BullBear_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components: need EMA13 on 6h high/low
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    ema_13_high = high_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_low = low_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema_13_high  # Bull Power = High - EMA13(high)
    bear_power = low - ema_13_low    # Bear Power = Low - EMA13(low)
    
    # Volume spike filter: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20)  # Need EMA34 and volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13_high[i]) or 
            np.isnan(ema_13_low[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power positive, price above 1d EMA34, volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative, price below 1d EMA34, volume spike
            elif (bear_power[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bear Power turns negative OR price crosses below 1d EMA34
            if (bear_power[i] < 0 or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive OR price crosses above 1d EMA34
            if (bull_power[i] > 0 or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals