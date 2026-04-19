#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h and 1d timeframe filters for trend direction and volatility regime.
# Uses 4h ADX to identify trending markets (ADX > 25) and 1d Bollinger Bands width percentile to filter low volatility (BBW < 50th percentile).
# Entry on 1h when price crosses above/below 20-period EMA with volume confirmation (volume > 1.5x 20-period average).
# Designed to work in both bull and bear markets by only taking trades in the direction of the 4h trend during low volatility periods.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
name = "1h_4h_ADX_1d_BBW_Volume_EMA_Cross"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for ADX calculation (trend strength)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX on 4h timeframe (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period]) if period > 1 else tr[0]
        plus_dm_smooth[period-1] = np.mean(plus_dm[1:period]) if period > 1 else plus_dm[0]
        minus_dm_smooth[period-1] = np.mean(minus_dm[1:period]) if period > 1 else minus_dm[0]
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period]) if 2*period-1 < len(dx) else 0
        
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Get 1d data for Bollinger Bands width calculation (volatility regime)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands on 1d timeframe (20-period, 2 std dev)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Calculate percentile of BB width (50th percentile = median)
    bb_width_percentile = np.zeros_like(bb_width)
    for i in range(len(bb_width)):
        if i < 20:
            bb_width_percentile[i] = np.nan
        else:
            bb_width_percentile[i] = np.percentile(bb_width[max(0, i-49):i+1], 50) if i >= 49 else np.percentile(bb_width[20:i+1], 50)
    
    # BB width < 50th percentile indicates low volatility regime
    bb_width_lower_than_median = bb_width < bb_width_percentile
    bb_width_filter_aligned = align_htf_to_ltf(prices, df_1d, bb_width_lower_than_median.astype(float))
    
    # Volume filter: volume > 1.5 * 20-period average (calculated on 1h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    # EMA filter: 20-period EMA on 1h
    ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(bb_width_filter_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Check 4h trend filter: ADX > 25 indicates trending market
        strong_trend = adx_4h_aligned[i] > 25
        
        # Check 1d volatility filter: low volatility regime (BBW < median)
        low_volatility = bb_width_filter_aligned[i] == 1.0
        
        # Only trade when both filters are active
        if strong_trend and low_volatility:
            if position == 0:
                # Long when price crosses above EMA with volume confirmation
                if close[i] > ema_20[i] and volume_filter[i]:
                    signals[i] = 0.20
                    position = 1
                # Short when price crosses below EMA with volume confirmation
                elif close[i] < ema_20[i] and volume_filter[i]:
                    signals[i] = -0.20
                    position = -1
                    
            elif position == 1:
                # Long position: exit when price crosses below EMA
                if close[i] < ema_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
                    
            elif position == -1:
                # Short position: exit when price crosses above EMA
                if close[i] > ema_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # If filters are not active, flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals