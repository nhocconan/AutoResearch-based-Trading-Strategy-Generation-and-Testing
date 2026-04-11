#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_vwap_bounce_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate daily VWAP
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_num_1d = np.cumsum(typical_price_1d * df_1d['volume'].values)
    vwap_den_1d = np.cumsum(df_1d['volume'].values)
    vwap_1d = vwap_num_1d / vwap_den_1d
    
    # Align VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Daily ATR for volatility filter
    high_low_1d = df_1d['high'].values - df_1d['low'].values
    high_close_1d = np.abs(df_1d['high'].values - df_1d['close'].values)
    low_close_1d = np.abs(df_1d['low'].values - df_1d['close'].values)
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h price relative to VWAP bands
    upper_band = vwap_1d_aligned + 0.5 * atr_1d_aligned
    lower_band = vwap_1d_aligned - 0.5 * atr_1d_aligned
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # VWAP bounce conditions
        touch_lower = price_close <= lower_band[i]
        touch_upper = price_close >= upper_band[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price touches or goes below VWAP - 0.5*ATR + volume confirmation
        if touch_lower and vol_confirm:
            enter_long = True
        
        # Short: Price touches or goes above VWAP + 0.5*ATR + volume confirmation
        if touch_upper and vol_confirm:
            enter_short = True
        
        # Exit conditions: price returns to VWAP
        exit_long = price_close >= vwap_1d_aligned[i]
        exit_short = price_close <= vwap_1d_aligned[i]
        
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

# Hypothesis: Daily VWAP acts as dynamic support/resistance. Price tends to revert to VWAP after
# deviations, especially with volume confirmation. Works in both bull and bear markets as VWAP
# adapts to price action. Bands at ±0.5*ATR provide entry zones with volatility adjustment.
# Position size 0.25 limits drawdown. Target: 30-80 trades/year.