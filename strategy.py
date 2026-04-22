#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel breakout with weekly EMA trend filter and volume confirmation.
# Works in bull markets (breakouts continue) and bear markets (breakouts fail quickly, limiting losses).
# Target: 10-25 trades/year on 1d timeframe to minimize fee drag.
# Uses weekly EMA for trend filter to avoid whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-day)
    upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to daily timeframe (no shift needed as already daily)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Daily ATR(14) for volatility filter and stop
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20  # Moderate volume surge
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume surge and above weekly EMA34
            if (close[i] > upper_aligned[i] and vol_surge[i] and close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with volume surge and below weekly EMA34
            elif (close[i] < lower_aligned[i] and vol_surge[i] and close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses opposite Donchian level or volatility drops significantly
            if position == 1:
                if close[i] < lower_aligned[i] or atr[i] < 0.5 * atr[i-1]:  # Volatility drop filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_aligned[i] or atr[i] < 0.5 * atr[i-1]:  # Volatility drop filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_DonchianBreakout_WeeklyEMA34_VolumeSurge"
timeframe = "1d"
leverage = 1.0