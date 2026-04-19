# 6h_Camarilla_R1_S1_Breakout_Volume_HTFTrend_V1
# Hypothesis: Camarilla pivot levels from daily timeframe provide key support/resistance.
# Breakout above R1 or below S1 with volume confirmation and aligned with 12h trend (EMA34) captures
# institutional flow. Works in bull/bear by filtering with higher timeframe trend.
# Target: 25-40 trades/year per symbol (~100-160 total over 4 years).

name = "6h_Camarilla_R1_S1_Breakout_Volume_HTFTrend_V1"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels from previous day
    # R4 = Close + (High-Low) * 1.1/2
    # R3 = Close + (High-Low) * 1.1/4
    # R2 = Close + (High-Low) * 1.1/6
    # R1 = Close + (High-Low) * 1.1/12
    # S1 = Close - (High-Low) * 1.1/12
    # S2 = Close - (High-Low) * 1.1/6
    # S3 = Close - (High-Low) * 1.1/4
    # S4 = Close - (High-Low) * 1.1/2
    camarilla_R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 6h (use previous day's levels for current day)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Calculate EMA34 on 12h for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure EMA34 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_R1_aligned[i]
        s1 = camarilla_S1_aligned[i]
        ema_34 = ema_34_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        if position == 0:
            # Enter long if price breaks above R1, above 12h EMA34 (uptrend), and volume confirmation
            if price > r1 and price > ema_34 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if price breaks below S1, below 12h EMA34 (downtrend), and volume confirmation
            elif price < s1 and price < ema_34 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below S1 (reversal) or trend changes
            if price < s1 or price < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above R1 (reversal) or trend changes
            if price > r1 or price > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf