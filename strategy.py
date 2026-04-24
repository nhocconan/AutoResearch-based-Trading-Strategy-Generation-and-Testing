#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and volume average calculation.
- Williams %R(14): Measures overbought/oversold levels (-20 to -80 range).
- Trend Filter: EMA34 on 1d timeframe - price above/below EMA determines bias.
- Volume Confirmation: Current volume > 2.0 * 20-period average volume on 1d.
- Entry: Long when Williams %R < -80 (oversold) AND price > 1d EMA34 AND volume spike.
         Short when Williams %R > -20 (overbought) AND price < 1d EMA34 AND volume spike.
- Exit: Williams %R crosses above -50 for long exit, below -50 for short exit.
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets by buying dips in uptrends, in bear markets by selling rallies in downtrends.
- Avoids choppy markets via trend filter requiring clear EMA34 alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Williams %R(14) on 12h timeframe
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # Neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34)  # Need 14 for Williams %R, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_wr = williams_r[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema34_1d_aligned[i]
        downtrend = curr_close < ema34_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: Williams %R crosses -50 midpoint
        if position != 0:
            # Exit long: Williams %R crosses above -50
            if position == 1:
                if curr_wr > -50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses below -50
            elif position == -1:
                if curr_wr < -50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extremes with trend and volume filters
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume confirmation
            long_condition = (curr_wr < -80 and 
                            uptrend and
                            volume_confirm)
            
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume confirmation
            short_condition = (curr_wr > -20 and 
                             downtrend and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0