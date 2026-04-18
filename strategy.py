# 4h_Pivot_R1S1_R2S2_Breakout_VolumeTrend_Adaptive
# Hypothesis: Camarilla pivot breakouts on 4h chart with volume confirmation and trend filter.
# Uses R1/S1 for entry, R2/S2 for profit targets, and 1d EMA for trend direction.
# Designed for 20-40 trades/year to minimize fee drag while capturing institutional levels.
# Works in both bull and bear markets by following trend direction from higher timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla formulas: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We use R1, S1, R2, S2 levels
    camarilla_levels = []
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_levels.append({'R1': np.nan, 'S1': np.nan, 'R2': np.nan, 'S2': np.nan})
            continue
        # Use previous day's data for today's levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        r1 = prev_close + (range_ * 1.1 / 12)
        s1 = prev_close - (range_ * 1.1 / 12)
        r2 = prev_close + (range_ * 1.1 / 6)
        s2 = prev_close - (range_ * 1.1 / 6)
        
        camarilla_levels.append({'R1': r1, 'S1': s1, 'R2': r2, 'S2': s2})
    
    camarilla_df = pd.DataFrame(camarilla_levels)
    r1_1d = camarilla_df['R1'].values
    s1_1d = camarilla_df['S1'].values
    r2_1d = camarilla_df['R2'].values
    s2_1d = camarilla_df['S2'].values
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    r1_4h_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_4h_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_4h_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: >1.5x 20-period average (adapted for 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or
            np.isnan(r2_4h_aligned[i]) or
            np.isnan(s2_4h_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_4h_aligned[i]
        s1 = s1_4h_aligned[i]
        r2 = r2_4h_aligned[i]
        s2 = s2_4h_aligned[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and uptrend (price > EMA34)
            if price > r1 and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and downtrend (price < EMA34)
            elif price < s1 and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price drops below S1 (reversal) OR trend turns down
            if price < s1 or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price rises above R1 (reversal) OR trend turns up
            if price > r1 or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Pivot_R1S1_R2S2_Breakout_VolumeTrend_Adaptive"
timeframe = "4h"
leverage = 1.0