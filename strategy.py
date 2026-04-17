# 12h_Camarilla_R1S1_Breakout_VolumeSpike_ADXFilter
# Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
# Breakout above R1 or below S1 with volume confirmation and ADX trend filter captures
# institutional order flow. Works in both bull/bear markets by only taking breakouts
# in direction of higher timeframe trend (ADX > 25).
# Target: 20-50 trades over 4 years (5-12/year) to minimize fee drag.

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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = Close + (High - Low) * 1.12 / 12
    # S1 = Close - (High - Low) * 1.12 / 12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + camarilla_range * 1.12 / 12
    s1 = close_1d - camarilla_range * 1.12 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate ADX(14) on 12h for trend strength filter
    # ADX requires +DM, -DM, TR
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    
    # Pad arrays to original length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    tr_smooth = wilders_smoothing(tr, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, period)
    
    # Volume spike filter: volume > 2x 20-period volume SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 34)  # need ADX and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
            
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        adx_val = adx[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        if position == 0:
            # Long: Breakout above R1 with volume spike and ADX > 25 (trending market)
            if price > r1_val and vol > 2.0 * vol_sma_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 with volume spike and ADX > 25
            elif price < s1_val and vol > 2.0 * vol_sma_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price re-enters below R1 or ADX weakens (< 20)
            if price < r1_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price re-enters above S1 or ADX weakens (< 20)
            if price > s1_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_VolumeSpike_ADXFilter"
timeframe = "12h"
leverage = 1.0