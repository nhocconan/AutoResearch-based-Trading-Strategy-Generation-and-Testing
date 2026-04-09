#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d ADX regime filter and volume confirmation
# - Uses 1d HTF for ADX regime: ADX > 25 = trending, ADX < 20 = ranging
# - In trending regime (ADX > 25): trade breakouts in direction of 1d EMA50 trend
# - In ranging regime (ADX < 20): mean revert at Camarilla H3/L3 levels
# - Volume confirmation: current 4h volume > 1.5x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_camarilla_adx_regime_v1"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMAs for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d ADX for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / np.where(atr == 0, 1, atr)
    di_minus = 100 * dm_minus_smooth / np.where(atr == 0, 1, atr)
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Calculate 1d Camarilla pivot levels from prior bar
    typical_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3_1d = typical_1d + range_1d * 1.1 / 4
    l3_1d = typical_1d - range_1d * 1.1 / 4
    h4_1d = typical_1d + range_1d * 1.1 / 2
    l4_1d = typical_1d - range_1d * 1.1 / 2
    
    # Align all 1d data to 4h timeframe (wait for completed 1d bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(h3_1d_aligned[i]) or
            np.isnan(l3_1d_aligned[i]) or np.isnan(h4_1d_aligned[i]) or
            np.isnan(l4_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        # Trend direction: 1d EMA50 > EMA200 = uptrend, < = downtrend
        uptrend = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        downtrend = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if trending:
                # In trending regime: exit when trend changes or price < EMA50
                if not uptrend or close[i] < ema_50_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:  # ranging regime
                # In ranging regime: exit when price > H3 (take profit) or < L3 (stop)
                if close[i] > h3_1d_aligned[i] or close[i] < l3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
                    
        elif position == -1:  # Short position
            # Exit conditions
            if trending:
                # In trending regime: exit when trend changes or price > EMA50
                if not downtrend or close[i] > ema_50_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:  # ranging regime
                # In ranging regime: exit when price < L3 (take profit) or > H3 (stop)
                if close[i] < l3_1d_aligned[i] or close[i] > h3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:  # Flat
            # Entry logic based on regime
            if volume_confirmed:
                if trending:
                    # In trending regime: trade breakouts in trend direction
                    if uptrend and close[i] > h4_1d_aligned[i]:
                        position = 1
                        signals[i] = position_size
                    elif downtrend and close[i] < l4_1d_aligned[i]:
                        position = -1
                        signals[i] = -position_size
                elif ranging:
                    # In ranging regime: mean revert at H3/L3
                    if close[i] > h3_1d_aligned[i]:
                        position = -1  # short at resistance
                        signals[i] = -position_size
                    elif close[i] < l3_1d_aligned[i]:
                        position = 1   # long at support
                        signals[i] = position_size
    
    return signals