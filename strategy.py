#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with volume confirmation and 1d trend filter
# - Long: price breaks above Camarilla R4, volume > 2.0x 20-period avg, 1d close > 1d EMA(50)
# - Short: price breaks below Camarilla S4, volume > 2.0x 20-period avg, 1d close < 1d EMA(50)
# - Exit: price returns to Camarilla pivot point (PP) or opposite Camarilla level (R3/S3)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Camarilla levels from 1d provide institutional support/resistance; breakouts with volume
#   and 1d trend alignment capture strong moves in both bull and bear markets

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla levels and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1 / 2
    # R3 = PP + (H - L) * 1.1 / 4
    # S3 = PP - (H - L) * 1.1 / 4
    # S4 = PP - (H - L) * 1.1 / 2
    typical_price = (high_1d + low_1d + close_1d) / 3
    hl_range = high_1d - low_1d
    camarilla_pp = typical_price
    camarilla_r4 = camarilla_pp + hl_range * 1.1 / 2
    camarilla_r3 = camarilla_pp + hl_range * 1.1 / 4
    camarilla_s3 = camarilla_pp - hl_range * 1.1 / 4
    camarilla_s4 = camarilla_pp - hl_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        pp = camarilla_pp_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Camarilla R4, volume confirmation, long bias
        if close_price > r4 and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below Camarilla S4, volume confirmation, short bias
        if close_price < s4 and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to Camarilla PP or drops to R3 (profit taking)
            exit_long = close_price <= pp or close_price <= r3
        elif position == -1:
            # Exit short if price returns to Camarilla PP or rises to S3 (profit taking)
            exit_short = close_price >= pp or close_price >= s3
        
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