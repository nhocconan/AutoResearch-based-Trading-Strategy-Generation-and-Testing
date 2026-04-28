#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based volume spike confirmation.
# Enter long when price breaks above Donchian upper band, 1d EMA34 trending up, and volume > 1.5x ATR-scaled average.
# Enter short when price breaks below Donchian lower band, 1d EMA34 trending down, and volume > 1.5x ATR-scaled average.
# Exit when price crosses the 1d EMA34 or reaches opposite Donchian band.
# Uses discrete position sizing (0.25) to minimize fee drag while maintaining profitability.
# Target: 80-150 total trades over 4 years (20-38/year) to avoid excessive fee churn.
# Donchian channels provide clear trend-following structure; 1d EMA34 filters for higher-timeframe trend alignment;
# ATR-scaled volume confirmation ensures breakouts have genuine momentum rather than random spikes.

name = "4h_Donchian20_1dEMA34_ATRVol_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR(14) for volume scaling and stoploss reference
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR-scaled average volume for confirmation
    # Volume > 1.5 * (20-period average volume) * (ATR / 20-period average ATR)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma_20 * (atr / atr_ma_20)
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper[i]
        breakout_down = close[i] < lower[i]
        
        # Exit conditions: price crosses 1d EMA34 or reaches opposite Donchian band
        exit_long = close[i] < ema_34_aligned[i] or close[i] < lower[i]
        exit_short = close[i] > ema_34_aligned[i] or close[i] > upper[i]
        
        # Handle entries and exits
        if breakout_up and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and ema_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals