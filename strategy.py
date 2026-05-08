#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Williams Alligator (13,8,5 SMAs) as trend filter, 4h Donchian(20) breakout, and volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment), price breaks above 4h Donchian upper band, volume > 1.8x average.
# Short when Alligator jaws > teeth > lips (bearish alignment), price breaks below 4h Donchian lower band, volume > 1.8x average.
# Includes ATR-based stop loss via signal=0 when price moves against position by 2.5x ATR.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.
# Williams Alligator identifies strong trends, Donchian captures breakouts, volume confirms conviction.

name = "4h_Alligator_Donchian_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator (SMAs)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Williams Alligator: Jaw (13-period), Teeth (8-period), Lips (5-period) SMAs
    jaw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_4h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_4h).rolling(window=5, min_periods=5).mean().values
    
    # Bullish alignment: jaws < teeth < lips
    # Bearish alignment: jaws > teeth > lips
    bullish_aligned = (jaw < teeth) & (teeth < lips)
    bearish_aligned = (jaw > teeth) & (teeth > lips)
    
    # Get 4h data for Donchian bands (same timeframe, but we still use get_htf_data for correctness)
    df_4h_dc = get_htf_data(prices, '4h')
    if len(df_4h_dc) < 20:
        return np.zeros(n)
    
    high_4h = df_4h_dc['high'].values
    low_4h = df_4h_dc['low'].values
    
    # 4h Donchian(20) bands
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h indicators to 4h (no actual alignment needed since same TF, but using helper for consistency)
    bullish_aligned = align_htf_to_ltf(prices, df_4h, bullish_aligned.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_4h, bearish_aligned.astype(float))
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h_dc, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h_dc, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # ATR for stop loss and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Position sizing: base 0.25, scaled by volatility (ATR/price)
    vol_factor = np.clip(atr / (close * 0.01), 0.5, 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 34  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(vol_factor[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish Alligator alignment, price breaks above Donchian upper band, volume spike
            if (bullish_aligned[i] > 0.5 and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.25 * vol_factor[i]
                position = 1
                entry_bar = i
            # Short: Bearish Alligator alignment, price breaks below Donchian lower band, volume spike
            elif (bearish_aligned[i] > 0.5 and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25 * vol_factor[i]
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: Bearish flip, price breaks below Donchian lower band, or stop loss (2.5*ATR)
            if (bearish_aligned[i] > 0.5 or 
                close[i] < donchian_low_aligned[i] or
                close[i] < close[entry_bar] - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 * vol_factor[i]
        elif position == -1:
            # Short exit: Bullish flip, price breaks above Donchian upper band, or stop loss (2.5*ATR)
            if (bullish_aligned[i] > 0.5 or 
                close[i] > donchian_high_aligned[i] or
                close[i] > close[entry_bar] + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 * vol_factor[i]
    
    return signals