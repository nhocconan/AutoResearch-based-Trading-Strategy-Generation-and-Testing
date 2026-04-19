# 4h_Camarilla_Pivot_Breakout_1dTrend_Volume
# Strategy: 4h Camarilla pivot breakout with daily trend filter and volume confirmation.
# Long when price breaks above R2 with price above 1d EMA34 and volume spike (>1.8x average).
# Short when price breaks below S2 with price below 1d EMA34 and volume spike.
# Uses 1d EMA34 as trend filter to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years).
name = "4h_Camarilla_Pivot_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla pivot and EMA34 calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R2, S2) from previous day
    pivot = (high_1d + low_1d + close_1d) / 3
    r2 = pivot + (high_1d - low_1d) * 1.1 / 4
    s2 = pivot - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate EMA34 on daily close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (wait for daily close)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above R2 AND above 1d EMA34
            if price > r2_val and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S2 AND below 1d EMA34
            elif price < s2_val and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below S2 or below 1d EMA34
            if price < s2_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above R2 or above 1d EMA34
            if price > r2_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals