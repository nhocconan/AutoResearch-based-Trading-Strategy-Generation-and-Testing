#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d ATR(14) volatility filter + 1d EMA34 trend + volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ATR(14) for volatility filter (ATR > 1.2 * 20-period ATR MA = high volatility regime).
         1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when price breaks above Donchian upper(20) AND uptrend AND high volatility AND volume > 1.5 * 4h volume MA(20);
         Short when price breaks below Donchian lower(20) AND downtrend AND high volatility AND volume > 1.5 * 4h volume MA(20).
- Exit: Long exits when price crosses below Donchian middle (10-period average of upper/lower);
        Short exits when price crosses above Donchian middle.
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian breakouts capture momentum; volatility filter avoids choppy regime losses; EMA34 filters higher-timeframe trend;
  volume spike confirms conviction. Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR(14) and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high + lowest_low) / 2.0
    
    # Get 4h data for volume MA(20)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 34)  # Donchian needs 20, volume MA needs 20, EMA34 needs 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volatility filter: ATR > 1.2 * 20-period ATR MA
        atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().values
        high_vol = atr_14_aligned[i] > 1.2 * atr_ma_20[i] if not np.isnan(atr_ma_20[i]) else False
        
        # Trend filter from 1d EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and high_vol and vol_confirm:
                # Long: price breaks above Donchian upper
                if curr_high > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and high_vol and vol_confirm:
                # Short: price breaks below Donchian lower
                if curr_low < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below Donchian middle
            if curr_close < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above Donchian middle
            if curr_close > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR14_VolFilter_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0