#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume confirmation + ADX regime filter
# Camarilla pivots from 1d provide intraday support/resistance levels that work well on 12h timeframe
# Volume confirmation ensures breakouts have conviction
# ADX > 25 filters for trending markets where breakouts are more reliable
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_camarilla_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # Formula: Close + (High - Low) * multiplier / 11
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate pivots for each 1d bar (using previous day's data to avoid look-ahead)
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Camarilla multipliers
    camarilla_multipliers = [1.1, 0.55, 0.275, 0.183]  # H1, H2, H3, H4 and L1-L4
    
    # Calculate resistance levels (H1-H4) and support levels (L1-L4)
    camarilla_h1 = prev_close_1d + (prev_high_1d - prev_low_1d) * camarilla_multipliers[0] / 11
    camarilla_h2 = prev_close_1d + (prev_high_1d - prev_low_1d) * camarilla_multipliers[1] / 11
    camarilla_h3 = prev_close_1d + (prev_high_1d - prev_low_1d) * camarilla_multipliers[2] / 11
    camarilla_h4 = prev_close_1d + (prev_high_1d - prev_low_1d) * camarilla_multipliers[3] / 11
    camarilla_l1 = prev_close_1d - (prev_high_1d - prev_low_1d) * camarilla_multipliers[0] / 11
    camarilla_l2 = prev_close_1d - (prev_high_1d - prev_low_1d) * camarilla_multipliers[1] / 11
    camarilla_l3 = prev_close_1d - (prev_high_1d - prev_low_1d) * camarilla_multipliers[2] / 11
    camarilla_l4 = prev_close_1d - (prev_high_1d - prev_low_1d) * camarilla_multipliers[3] / 11
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (20-period)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (14-period)
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, dm_plus_smooth / atr_1d * 100, 0)
    di_minus = np.where(atr_1d != 0, dm_minus_smooth / atr_1d * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar close)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h1_aligned[i]) or np.isnan(camarilla_l1_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 1d average volume
        volume_confirmed = volume[i] > 1.3 * avg_volume_1d_aligned[i]
        
        # Regime filter: ADX > 25 = trending (follow breakout), ADX <= 25 = ranging (mean revert)
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] <= 25
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR regime shifts to ranging
            if close[i] < camarilla_l3_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR regime shifts to ranging
            if close[i] > camarilla_h3_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime and volume_confirmed:
                # Follow breakout in trending regime
                if close[i] > camarilla_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < camarilla_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime and volume_confirmed:
                # Mean revert at Camarilla H4/L4 in ranging regime
                if close[i] < camarilla_l4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > camarilla_h4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals