# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: 12h Camarilla pivot (R1/S1) breakout with 1d EMA(34) trend filter and volume confirmation.
# Uses daily pivot levels for structure, daily EMA for trend direction, and volume spike for confirmation.
# Long when price breaks above R1 in uptrend (close > daily EMA34) with volume spike.
# Short when price breaks below S1 in downtrend (close < daily EMA34) with volume spike.
# Exit on opposite pivot touch or trend reversal.
# Designed for 12h timeframe to target 12-37 trades/year per symbol.
# Works in bull/bear via trend filter + volatility-based entry levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for pivot and trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    hl_range = high_1d - low_1d
    r1 = close_1d + 1.1 * hl_range / 12
    s1 = close_1d - 1.1 * hl_range / 12
    
    # 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe (waits for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + uptrend (close > EMA34) + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend (close < EMA34) + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on S1 touch or trend reversal
                if (close[i] < s1_aligned[i] or close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on R1 touch or trend reversal
                if (close[i] > r1_aligned[i] or close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0