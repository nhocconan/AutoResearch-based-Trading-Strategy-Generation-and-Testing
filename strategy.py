#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d ADX regime filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold) in bullish regime (ADX>25) with volume spike
# Short when Williams %R crosses below -20 (overbought) in bearish regime (ADX>25) with volume spike
# Uses 1d ADX to filter for trending markets only, avoiding whipsaws in ranging conditions
# Volume confirmation ensures reversals have institutional participation
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag

name = "4h_WilliamsR_1dADX25_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[1:period+1])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smooth(dx, 14)
    
    # Align daily ADX to 4h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams %R(14) on 4h
    williams_window = 14
    highest_high = pd.Series(high).rolling(window=williams_window, min_periods=williams_window).max().values
    lowest_low = pd.Series(low).rolling(window=williams_window, min_periods=williams_window).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * (highest_high - close) / (highest_high - lowest_low), 
                          -50)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, williams_window)  # warmup for ADX and Williams %R
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams = williams_r[i]
        curr_adx = adx_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Regime filter: only trade in trending markets (ADX > 25)
        is_trending = curr_adx > 25
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in trending regime
            if is_trending and curr_volume_confirm:
                # Bullish reversal: Williams %R crosses above -80 from below
                if curr_williams > -80 and (i == start_idx or williams_r[i-1] <= -80):
                    signals[i] = 0.25
                    position = 1
                # Bearish reversal: Williams %R crosses below -20 from above
                elif curr_williams < -20 and (i == start_idx or williams_r[i-1] >= -20):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Williams %R crosses above -20 (overbought) OR crosses below -80 with volume
            if curr_williams > -20 or (curr_williams < -80 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Williams %R crosses below -80 (oversold) OR crosses above -20 with volume
            if curr_williams < -80 or (curr_williams > -20 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals