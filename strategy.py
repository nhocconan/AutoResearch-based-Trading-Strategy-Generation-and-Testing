#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Uses weekly Camarilla levels (R3/S3, R4/S4) from 1w timeframe
# - Long: price breaks above R4 with volume > 2x 20-period average and price > 1w EMA(50)
# - Short: price breaks below S4 with volume > 2x 20-period average and price < 1w EMA(50)
# - Exit: price returns to 1w Camarilla pivot point (PP) or opposite S3/R3 level
# - Volume confirmation prevents false breakouts in low-liquidity periods
# - Weekly EMA(50) filter ensures alignment with higher-timeframe trend
# - Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
# - Works in both bull and bear markets by capturing institutional breakout attempts

name = "6h_1w_camarilla_breakout_volume_v1"
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
    
    # Load 1w data ONCE before loop for Camarilla levels and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w Camarilla levels (based on previous week's OHLC)
    # Camarilla formula: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # PP = (high + low + close) / 3
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each 1w bar
    rng = high_1w - low_1w
    r4_1w = close_1w + (rng * 1.1 / 2)
    r3_1w = close_1w + (rng * 1.1 / 4)
    pp_1w = (high_1w + low_1w + close_1w) / 3
    s3_1w = close_1w - (rng * 1.1 / 4)
    s4_1w = close_1w - (rng * 1.1 / 2)
    
    # Align 1w Camarilla levels to 6h timeframe (wait for weekly bar close)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Pre-compute 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(pp_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(s4_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels from aligned arrays
        r4 = r4_1w_aligned[i]
        r3 = r3_1w_aligned[i]
        pp = pp_1w_aligned[i]
        s3 = s3_1w_aligned[i]
        s4 = s4_1w_aligned[i]
        
        # Volume confirmation: current volume > 2x 20-period average (strict filter)
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # 1w EMA trend bias
        ema_bias_long = close_price > ema_50_1w_aligned[i]
        ema_bias_short = close_price < ema_50_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above R4 with volume confirmation and long bias
        if close_price > r4 and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below S4 with volume confirmation and short bias
        if close_price < s4 and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point or drops below S3
            exit_long = close_price <= pp or close_price < s3
        elif position == -1:
            # Exit short if price returns to pivot point or rises above R3
            exit_short = close_price >= pp or close_price > r3
        
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