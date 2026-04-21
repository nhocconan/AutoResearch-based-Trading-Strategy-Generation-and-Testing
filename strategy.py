#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ChopRegime_v3
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike (>2.0), and chop regime filter (CHOP > 61.8 = range, < 38.2 = trending). In ranging markets (CHOP > 61.8), fade breaks of R1/S1 with mean reversion to H5. In trending markets (CHOP < 38.2), breakout of R1/S1 continues trend. Designed for low trade frequency (~20-40/year) with regime adaptation to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === ATR for stoploss (14-period on 4h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Choppiness Index (14-period on 4h) ===
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (highest_14 - lowest_14)) / np.log10(14)
    chop[np.isnan(chop)] = 50.0  # default to neutral when undefined
    
    # === Camarilla levels from previous 1d bar (OHLC) ===
    # We need the previous completed 1d bar's OHLC to compute today's levels
    # Since we're using 1d HTF data aligned, we can use the previous day's values
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R1 = close_1d + (high_1d - low_1d) * 1.0/12
    S1 = close_1d - (high_1d - low_1d) * 1.0/12
    H5 = (high_1d + low_1d) / 2  # midpoint (H5)
    
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    H5_aligned = align_htf_to_ltf(prices, df_1d, H5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(H5_aligned[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        h5 = H5_aligned[i]
        chop_val = chop[i]
        atr_val = atr_14[i]
        
        # Regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (follow breakout)
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long conditions
            long_breakout = price_close > r1 and vol_spike > 2.0
            long_mean_revert = price_close < s1 and vol_spike > 2.0 and price_close > h5
            
            # Short conditions
            short_breakout = price_close < s1 and vol_spike > 2.0
            short_mean_revert = price_close > r1 and vol_spike > 2.0 and price_close < h5
            
            if is_trending:
                # In trending market: follow breakout
                if long_breakout and price_close > trend_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price_close
                elif short_breakout and price_close < trend_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price_close
            elif is_ranging:
                # In ranging market: mean revert at extremes
                if long_mean_revert:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price_close
                elif short_mean_revert:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price_close
        
        elif position != 0:
            # Exit conditions: ATR trailing stop or opposite signal
            if position == 1:
                # Long exit: price drops 2.0 * ATR below entry OR breaks S1 with volume
                if price_close <= entry_price - 2.0 * atr_val or (price_close < s1 and vol_spike > 1.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Short exit: price rises 2.0 * ATR above entry OR breaks R1 with volume
                if price_close >= entry_price + 2.0 * atr_val or (price_close > r1 and vol_spike > 1.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ChopRegime_v3"
timeframe = "4h"
leverage = 1.0