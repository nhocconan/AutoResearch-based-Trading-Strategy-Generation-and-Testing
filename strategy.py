#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d volume spike and ADX regime filter
# In low volatility regimes (ADX < 25): mean reversion at Williams %R extremes (>80 oversold, <20 overbought) with volume confirmation
# In high volatility regimes (ADX >= 25): trend following on Williams %R cross of -50 with volume confirmation
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Williams %R identifies exhaustion points, volume confirms conviction, ADX adapts to market regime
# Works in bull/bear markets: mean reversion captures bounces in ranging markets, trend following catches momentum in trending markets

name = "6h_1d_williamsr_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for ADX calculation
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate +DM and -DM for ADX
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Calculate +DI and -DI
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    
    # Calculate DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Calculate 1d Williams %R (14-period)
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)
    
    # Calculate 1d average volume (20-period)
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r_1d_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Regime filter based on ADX
        low_vol_regime = adx_1d_aligned[i] < 25
        high_vol_regime = adx_1d_aligned[i] >= 25
        
        if position == 1:  # Long position
            if low_vol_regime:
                # Exit long if Williams %R rises above -20 (overbought) or regime shifts to high vol
                if williams_r_1d_aligned[i] > -20 or high_vol_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif high_vol_regime:
                # Exit long if Williams %R falls below -50 or regime shifts to low vol
                if williams_r_1d_aligned[i] < -50 or low_vol_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if low_vol_regime:
                # Exit short if Williams %R falls below -80 (oversold) or regime shifts to high vol
                if williams_r_1d_aligned[i] < -80 or high_vol_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif high_vol_regime:
                # Exit short if Williams %R rises above -50 or regime shifts to low vol
                if williams_r_1d_aligned[i] > -50 or low_vol_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if low_vol_regime:
                # Mean reversion: enter long when Williams %R < -80 (oversold) with volume confirmation
                # Enter short when Williams %R > -20 (overbought) with volume confirmation
                if williams_r_1d_aligned[i] < -80 and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif williams_r_1d_aligned[i] > -20 and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif high_vol_regime:
                # Trend following: enter long when Williams %R crosses above -50 with volume confirmation
                # Enter short when Williams %R crosses below -50 with volume confirmation
                if williams_r_1d_aligned[i] > -50 and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif williams_r_1d_aligned[i] < -50 and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals