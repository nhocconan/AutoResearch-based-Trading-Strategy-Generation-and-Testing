#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels (R4/S3) with 1d volume confirmation
# - Long when price breaks above weekly R4 level with 1d volume spike and 1d close > open (bullish candle)
# - Short when price breaks below weekly S3 level with 1d volume spike and 1d close < open (bearish candle)
# - Uses weekly pivot points calculated from prior week's OHLC, providing structural support/resistance
# - Volume confirmation ensures breakout strength; daily candle direction adds momentum filter
# - Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 6h timeframe
# - Discrete position sizing (0.25) reduces churn; ATR-based stop manages risk
# - Weekly timeframe provides stable levels less prone to whipsaw than daily pivots

name = "6h_1w_camarilla_r4s3_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: > 2.0x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d bullish/bearish candle
    bullish_candle_1d = close_1d > open_1d
    bearish_candle_1d = close_1d < open_1d
    bullish_candle_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_candle_1d)
    bearish_candle_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_candle_1d)
    
    # Weekly Camarilla pivot levels (R4, S3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Calculate Camarilla levels
    r4_1w = pp_1w + ((high_1w - low_1w) * 1.1 / 2.0)
    s3_1w = pp_1w - ((high_1w - low_1w) * 1.1 / 4.0)
    
    # Align weekly levels to 6h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # 6h ATR(14) for stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_6h = np.zeros_like(tr)
    atr_14_6h[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_6h[i] = (atr_14_6h[i-1] * (14-1) + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(bullish_candle_1d_aligned[i]) or 
            np.isnan(bearish_candle_1d_aligned[i]) or np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price breaks below S3
            if (prices['close'].iloc[i] < entry_price - 2.5 * entry_atr or 
                prices['close'].iloc[i] < s3_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price breaks above R4
            if (prices['close'].iloc[i] > entry_price + 2.5 * entry_atr or 
                prices['close'].iloc[i] > r4_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout with volume and candle direction filters
            if vol_spike_1d_aligned[i]:
                # Long signal: price breaks above weekly R4 with bullish daily candle
                if (prices['close'].iloc[i] > r4_1w_aligned[i] and 
                    bullish_candle_1d_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_6h[i]
                    signals[i] = 0.25
                # Short signal: price breaks below weekly S3 with bearish daily candle
                elif (prices['close'].iloc[i] < s3_1w_aligned[i] and 
                      bearish_candle_1d_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_6h[i]
                    signals[i] = -0.25
    
    return signals