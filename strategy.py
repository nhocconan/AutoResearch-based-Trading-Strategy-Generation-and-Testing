#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX regime filter and volume spike confirmation
# - Uses 1d HTF to calculate ADX(14) for trend strength regime (ADX > 25 = trending)
# - Williams %R(14) on 6h: long when crosses above -80 from below, short when crosses below -20 from above
# - Volume confirmation: current 6h volume > 2.0x 20-period average to filter low-momentum breakouts
# - Fixed position size 0.25 to control drawdown
# - Williams %R captures overbought/oversold reversals; ADX ensures we only trade in trending markets
# - Volume spike confirms institutional participation
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)

name = "6h_1d_williamsr_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for trend regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # First bar
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, dm_plus_smooth / atr_1d * 100, 0)
    di_minus = np.where(atr_1d != 0, dm_minus_smooth / atr_1d * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R(14) on 6h
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0,
                          -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14),
                          -50)  # Neutral when range=0
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        trending_regime = adx_aligned[i] > 25
        
        # Volume confirmation: current 6h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:  # Flat - look for entry
            if trending_regime and volume_confirmed:
                # Williams %R entry signals
                # Long: Williams %R crosses above -80 from below (oversold bounce in uptrend)
                if i > 0 and williams_r[i-1] <= -80 and williams_r[i] > -80:
                    position = 1
                    signals[i] = 0.25
                # Short: Williams %R crosses below -20 from above (overbought rejection in downtrend)
                elif i > 0 and williams_r[i-1] >= -20 and williams_r[i] < -20:
                    position = -1
                    signals[i] = -0.25
        else:  # Position open - look for exit
            # Exit when Williams %R returns to neutral territory (-50) or reverses
            if position == 1 and williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals