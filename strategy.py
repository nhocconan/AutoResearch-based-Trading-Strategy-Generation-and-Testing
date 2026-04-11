#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Pre-compute weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Pre-compute weekly ATR for volatility filter
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[low_1w[0]], low_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Pre-compute daily volume SMA for confirmation
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = price_high > donchian_high_aligned[i]
        breakout_down = price_low < donchian_low_aligned[i]
        
        # Volatility filter: current ATR > 1.5x weekly ATR
        vol_filter = atr_1w_aligned[i] > 0 and (price_high - price_low) > 1.5 * atr_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-day average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Upward breakout + volatility filter + volume confirmation
        if breakout_up and vol_filter and vol_confirm:
            enter_long = True
        
        # Short: Downward breakout + volatility filter + volume confirmation
        if breakout_down and vol_filter and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite breakout or volatility collapse
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on downward breakout or volatility collapse
            exit_long = breakout_down or not vol_filter
        elif position == -1:
            # Exit short on upward breakout or volatility collapse
            exit_short = breakout_up or not vol_filter
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
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