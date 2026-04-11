#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_volatility_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels with proper min_periods
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed 1w bars (no look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    donchian_high[0] = np.nan
    donchian_low[0] = np.nan
    
    # Align 1w Donchian levels to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # 6h ATR for volatility filter and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: volume > 1.8x 30-period average (selective)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_30[i]
        atr_val = atr[i]
        
        # Volume confirmation: selective threshold
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.008 * price_close  # ATR > 0.8% of price
        
        # Long conditions: price breaks above 1w Donchian high with volume and vol filter
        long_signal = volume_confirmed and vol_filter and (price_high > donchian_high_6h[i])
        
        # Short conditions: price breaks below 1w Donchian low with volume and vol filter
        short_signal = volume_confirmed and vol_filter and (price_low < donchian_low_6h[i])
        
        # Exit when price returns to midpoint of 1w Donchian channel
        donchian_mid = (donchian_high_6h[i] + donchian_low_6h[i]) / 2
        exit_long = position == 1 and price_close < donchian_mid
        exit_short = position == -1 and price_close > donchian_mid
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25  # Size: 25%
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1w Donchian breakouts with volume confirmation capture strong trends
# while volatility filter prevents whipsaws in low-volatility environments.
# Enters long when 6h price breaks above 1w Donchian high (20-period) with
# volume > 1.8x average and ATR > 0.8% of price.
# Enters short when price breaks below 1w Donchian low with same conditions.
# Exits when price returns to midpoint of the Donchian channel.
# Works in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 15-25 trades/year to minimize fee drag while capturing major moves.