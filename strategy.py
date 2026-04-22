#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout + 1w EMA14 trend filter + volume confirmation
    # Donchian breakout captures breakout moves in both bull and bear markets
    # 1w EMA14 filter ensures alignment with longer-term trend to avoid counter-trend trades
    # Volume confirmation (1.5x 20-period MA) filters for institutional participation
    # Works in bull/bear: breakouts with volume in trend direction capture sustained moves
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for EMA14 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema14_1w = pd.Series(df_1w['close']).ewm(span=14, adjust=False, min_periods=14).mean().values
    ema14_1w_aligned = align_htf_to_ltf(prices, df_1w, ema14_1w)
    
    # Donchian channels on 1d (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema14_1w_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above Donchian high + volume + price above 1w EMA14
            if close[i] > donch_high[i] and vol_confirm[i] and close[i] > ema14_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low + volume + price below 1w EMA14
            elif close[i] < donch_low[i] and vol_confirm[i] and close[i] < ema14_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level (mean reversion within channel)
            if position == 1:
                if close[i] < donch_low[i]:  # Price breaks below lower band
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donch_high[i]:  # Price breaks above upper band
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian_20_Breakout_1wEMA14_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0