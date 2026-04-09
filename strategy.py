#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX regime filter
# In strong trending regimes (1d ADX > 25): Williams %R mean reversion (long < -80, short > -20)
# In weak/range regimes (1d ADX <= 25): Williams %R trend following (long > -20, short < -80)
# Volume confirmation on entries to reduce false signals
# Works in bull/bear markets: adapts to regime, avoids whipsaws in ranging markets, captures trends
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_1d_williamsr_adx_regime_v1"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = np.where(
        (highest_high_1d - lowest_low_1d) != 0,
        ((highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)) * -100,
        -50  # neutral when range is zero
    )
    
    # Calculate 1d ADX (14-period) for trend strength
    def calculate_atr(high, low, close, period):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
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
        
        atr = wilders_smoothing(tr, period)
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # +DM and -DM
    up_move = np.concatenate([[np.nan], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[np.nan], low_1d[:-1] - low_1d[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth +DM, -DM, and TR
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    tr_smooth = wilders_smoothing(np.concatenate([[np.nan], 
                                                  np.absolute(high_1d[1:] - low_1d[1:]),
                                                  np.absolute(high_1d[1:] - close_1d[:-1]),
                                                  np.absolute(low_1d[1:] - close_1d[:-1])]), 14)
    # Fix TR smoothing - recalculate properly
    tr_calc = np.concatenate([[np.nan], 
                              np.maximum(np.abs(high_1d[1:] - low_1d[1:]),
                                         np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                                    np.abs(low_1d[1:] - close_1d[:-1])))])
    tr_smooth = wilders_smoothing(tr_calc, 14)
    
    # +DI and -DI
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 
                  0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 6h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        williams_r = williams_r_1d_aligned[i]
        adx = adx_1d_aligned[i]
        
        if position == 1:  # Long position
            if adx > 25:  # Strong trend - mean reversion
                # Exit long if Williams %R rises above -20 (overbought)
                if williams_r > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Weak trend/range - trend following
                # Exit long if Williams %R falls below -80 (oversold) or reverses
                if williams_r < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            if adx > 25:  # Strong trend - mean reversion
                # Exit short if Williams %R falls below -80 (oversold)
                if williams_r < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Weak trend/range - trend following
                # Exit short if Williams %R rises above -20 (overbought) or reverses
                if williams_r > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if adx > 25:  # Strong trend - mean reversion entries
                # Enter long when Williams %R < -80 (oversold) with volume confirmation
                if williams_r < -80 and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short when Williams %R > -20 (overbought) with volume confirmation
                elif williams_r > -20 and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            else:  # Weak trend/range - trend following entries
                # Enter long when Williams %R > -20 (overbought) with volume confirmation
                if williams_r > -20 and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short when Williams %R < -80 (oversold) with volume confirmation
                elif williams_r < -80 and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals