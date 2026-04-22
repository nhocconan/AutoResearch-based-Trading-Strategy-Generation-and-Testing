#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA34 trend filter and volume spike
# Williams %R measures overbought/oversold levels: -20 to -80 range
# Long when Williams %R crosses above -80 from below + 1d uptrend + volume spike
# Short when Williams %R crosses below -20 from above + 1d downtrend + volume spike
# Mean reversion in ranging markets with trend filter to avoid false signals in strong trends
# Designed for 4h timeframe targeting 20-40 trades/year per symbol.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R crossovers
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]
    
    # Cross above -80 (oversold to normal)
    cross_above_80 = (williams_r > -80) & (williams_r_prev <= -80)
    # Cross below -20 (overbought to normal)
    cross_below_20 = (williams_r < -20) & (williams_r_prev >= -20)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(williams_r_prev[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 + 1d uptrend + volume spike
            if (cross_above_80[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 + 1d downtrend + volume spike
            elif (cross_below_20[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R reverts to opposite extreme or trend reversal
            if position == 1:
                # Exit on Williams %R crossing below -20 (overbought) or trend reversal
                if (williams_r[i] < -20 or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Williams %R crossing above -80 (oversold) or trend reversal
                if (williams_r[i] > -80 or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0