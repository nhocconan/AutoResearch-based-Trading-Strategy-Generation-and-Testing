#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume confirmation and chop regime filter
# Uses tighter volume confirmation (2.0x avg volume) and stricter chop thresholds to reduce trades
# In trending regime (low chop): breakout strategy at H3/L3 levels
# In ranging regime (high chop): mean reversion at H4/L4 levels
# Discrete position sizing 0.25 to target ~20-50 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows trends in trending regimes, mean reversion at pivots in ranging regimes

name = "4h_1d_camarilla_breakout_v2"
timeframe = "4h"
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.zeros_like(close_1d)
    
    # Calculate 1d Camarilla pivot levels
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d
    camarilla_l3 = close_1d - 1.1 * range_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    
    # Calculate 1d ATR(10) for volume avg reference
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
    
    atr_10_1d = wilders_smoothing(tr, 10)
    
    # Calculate 1d average volume (20-period)
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Bollinger Band Width for chop regime filter (using 1d data)
    close_s_1d = pd.Series(close_1d)
    basis_1d = close_s_1d.rolling(window=20, min_periods=20).mean().values
    dev_1d = close_s_1d.rolling(window=20, min_periods=20).std().values
    upper_bb_1d = basis_1d + 2.0 * dev_1d
    lower_bb_1d = basis_1d - 2.0 * dev_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / basis_1d
    bb_width_1d = np.where(basis_1d != 0, bb_width_1d, 0)
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(bb_width_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume (stricter)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Chop regime: stricter thresholds to reduce trades
        trending_regime = bb_width_1d_aligned[i] < 0.03  # Very low volatility = trending
        ranging_regime = bb_width_1d_aligned[i] > 0.08   # High volatility = ranging
        
        if position == 1:  # Long position
            if trending_regime and volume_confirmed:
                # Exit long if price falls below H3
                if close[i] < camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price moves back above L4 (mean reversion exit)
                if close[i] > camarilla_l4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime and volume_confirmed:
                # Exit short if price rises above L3
                if close[i] > camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price moves back below H4 (mean reversion exit)
                if close[i] < camarilla_h4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime and volume_confirmed:
                # Breakout strategy in trending market
                if close[i] > camarilla_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < camarilla_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion at extremes in ranging market
                if close[i] < camarilla_l4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > camarilla_h4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals