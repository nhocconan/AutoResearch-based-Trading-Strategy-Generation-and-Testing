#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with volume confirmation and 12h trend filter
# - Long: price breaks above Camarilla R4, volume > 2x 20-period avg, 12h close > EMA(50)
# - Short: price breaks below Camarilla S4, volume > 2x 20-period avg, 12h close < EMA(50)
# - Exit: price returns to Camarilla pivot point (PP) or opposite Camarilla level (R3/S3)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Camarilla levels derived from prior 12h bar (completed bar) for no look-ahead
# - Works in both trending and ranging markets by capturing institutional breakouts

name = "6h_12h_camarilla_breakout_v1"
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
    
    # Load 12h data ONCE before loop for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-compute Camarilla levels from completed 12h bar
    # Formula based on prior 12h bar's high, low, close
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # R3 = PP + (H - L) * 1.1/4
    # S3 = PP - (H - L) * 1.1/4
    # S4 = PP - (H - L) * 1.1/2
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pp = (high_12h + low_12h + close_12h) / 3.0
    r4 = pp + (high_12h - low_12h) * 1.1 / 2.0
    r3 = pp + (high_12h - low_12h) * 1.1 / 4.0
    s3 = pp - (high_12h - low_12h) * 1.1 / 4.0
    s4 = pp - (high_12h - low_12h) * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe (available after 12h bar closes)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # 12h EMA trend filter
        ema_bias_long = close_price > ema_50_12h_aligned[i]
        ema_bias_short = close_price < ema_50_12h_aligned[i]
        
        # Camarilla levels
        r4_level = r4_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        s4_level = s4_aligned[i]
        pp_level = pp_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above R4 with volume confirmation and long bias
        if close_price > r4_level and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price closes below S4 with volume confirmation and short bias
        if close_price < s4_level and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point or breaks below R3 (failed breakout)
            exit_long = close_price <= pp_level or close_price < r3_level
        elif position == -1:
            # Exit short if price returns to pivot point or breaks above S3 (failed breakout)
            exit_short = close_price >= pp_level or close_price > s3_level
        
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