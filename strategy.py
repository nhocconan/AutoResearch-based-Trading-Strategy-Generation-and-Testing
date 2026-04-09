#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels with volume confirmation
# In low volatility regimes (choppy markets): fade at R3/S3 levels for mean reversion
# In high volatility regimes (trending markets): breakout continuation at R4/S4 levels
# Uses 1d ADX to filter regimes and 12h volume spike for confirmation
# Discrete position sizing 0.25 to limit trades to 12-37/year and reduce fee drag
# Works in bull/bear markets: mean reversion in ranging, breakout in trending

name = "6h_12h_1d_camarilla_adx_volume_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Camarilla: R4 = Close + ((High - Low) * 1.1/2), R3 = Close + ((High - Low) * 1.1/4)
    #          S3 = Close - ((High - Low) * 1.1/4), S4 = Close - ((High - Low) * 1.1/2)
    # We use the previous bar's levels to avoid look-ahead
    high_shift = np.concatenate([[np.nan], high_12h[:-1]])
    low_shift = np.concatenate([[np.nan], low_12h[:-1]])
    close_shift = np.concatenate([[np.nan], close_12h[:-1]])
    
    rangep = high_shift - low_shift
    r4 = close_shift + rangep * 1.1 / 2
    r3 = close_shift + rangep * 1.1 / 4
    s3 = close_shift - rangep * 1.1 / 4
    s4 = close_shift - rangep * 1.1 / 2
    
    # Calculate 12h volume spike (volume > 1.5 * 20-period average)
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (vol_ma * 1.5)
    
    # Load 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for regime filtering
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[:-1])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = wilders_smoothing(tr, period)
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smoothing(dx, period)
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 12h indicators to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(volume_spike_12h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 1d ADX
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below R3 in trending market
                if close[i] <= r3_12h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price returns above S3 in ranging market
                if close[i] >= s3_12h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above S3 in trending market
                if close[i] >= s3_12h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price returns below R3 in ranging market
                if close[i] <= r3_12h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime and volume_spike_12h_aligned[i]:
                # Breakout continuation in trending market with volume confirmation
                if close[i] > r4_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion at extreme levels in ranging market
                if close[i] < s3_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > r3_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals