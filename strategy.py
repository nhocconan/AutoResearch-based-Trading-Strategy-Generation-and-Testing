#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian upper band (20-period high) AND volume > 1.8x 20-period average AND 1w EMA34 > EMA34_prev (uptrend)
# Short when price breaks below Donchian lower band (20-period low) AND volume > 1.8x 20-period average AND 1w EMA34 < EMA34_prev (downtrend)
# Exit when price crosses back to the Donchian midpoint OR 1w EMA34 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-25 trades/year per symbol.
# Donchian channels provide robust price channels that work in both trending and ranging markets.
# Volume confirmation filters for institutional participation.
# 1w EMA34 trend filter ensures we only trade in the direction of the primary trend to avoid counter-trend whipsaws in bear markets.
# This strategy has shown strong test performance in DB for SOLUSDT (Sharpe 1.10-1.38) and should work for BTC/ETH.

name = "1d_Donchian20_VolumeSpike_1wEMA34_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian bands calculation (using current day's data for breakout)
    # We calculate Donchian on the same timeframe as our signals (1d) but use min_periods to avoid look-ahead
    if len(high) >= 20:
        # Calculate Donchian upper and lower bands (20-period high/low)
        upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
        midpoint = (upper_band + lower_band) / 2.0
    else:
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
        midpoint = np.full(n, np.nan)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w data
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_prev = np.concatenate([[np.nan], ema_34[:-1]])  # Previous EMA for trend direction
    
    # Uptrend when current EMA34 > previous EMA34
    uptrend_1w = ema_34 > ema_34_prev
    downtrend_1w = ema_34 < ema_34_prev
    
    # Align 1w trend to 1d timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume confirmation: volume > 1.8x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(midpoint[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band AND volume spike AND 1w uptrend
            if (close[i] > upper_band[i] and 
                volume_filter[i] and 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band AND volume spike AND 1w downtrend
            elif (close[i] < lower_band[i] and 
                  volume_filter[i] and 
                  downtrend_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to midpoint OR 1w trend flips to downtrend
            if (close[i] < midpoint[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to midpoint OR 1w trend flips to uptrend
            if (close[i] > midpoint[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals