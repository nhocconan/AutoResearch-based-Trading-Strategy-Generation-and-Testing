# 12h_Camarilla_R1S1_Touch_1dEMA34_Volume
# Hypothesis: Price tends to revert to 1-day EMA34 after touching 1-day Camarilla R1/S1 levels.
# In bull/bear markets, price often retraces to trend (EMA34) after testing key pivot levels.
# Long when price touches S1 from above and closes above EMA34 with volume confirmation.
# Short when price touches R1 from below and closes below EMA34 with volume confirmation.
# Uses 1d EMA34 as trend filter and 1d volume spike for confirmation.
# Target: 12-37 trades/year per symbol (~50-150 total over 4 years).
# Timeframe: 12h (lower frequency reduces fee drift).

name = "12h_Camarilla_R1S1_Touch_1dEMA34_Volume"
timeframe = "12h"
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
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = pivot + (high_1d - low_1d) * 1.1 / 4
    s1 = pivot - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate EMA34 on daily close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe (wait for daily close)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current 12h volume > 1.5x 20-period average of 12h volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price touches S1 from above AND closes above EMA34
            if low[i] <= s1_val and close[i] > s1_val and close[i] > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price touches R1 from below AND closes below EMA34
            elif high[i] >= r1_val and close[i] < r1_val and close[i] < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price closes below EMA34 or touches R1
            if close[i] < ema_trend or close[i] >= r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price closes above EMA34 or touches S1
            if close[i] > ema_trend or close[i] <= s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals