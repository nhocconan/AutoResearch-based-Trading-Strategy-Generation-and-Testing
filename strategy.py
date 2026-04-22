#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1h 4-hour Donchian breakout with 1-day EMA trend filter and volume confirmation.
    # Works in bull/bear markets: breakouts capture directional moves, EMA filters trend direction.
    # Uses 4h for signal direction, 1h for entry timing to reduce false signals.
    
    # Load 4h data once
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe (wait for 4h bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, high_max_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, low_min_20)
    
    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Volume filter (20-period surge on 1h)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Price array
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_1d_34_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high with volume surge AND 1d EMA34 uptrend
            if close[i] > donchian_high_aligned[i] and vol_surge[i] and close[i] > ema_1d_34_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian low with volume surge AND 1d EMA34 downtrend
            elif close[i] < donchian_low_aligned[i] and vol_surge[i] and close[i] < ema_1d_34_aligned[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < donchian_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > donchian_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian_Breakout_1dEMA34_Trend_VolumeSurge_Session_v1"
timeframe = "1h"
leverage = 1.0