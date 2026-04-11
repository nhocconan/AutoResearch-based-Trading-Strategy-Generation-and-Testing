#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h trend filter and volume confirmation
# - Uses 12h timeframe for trend direction (EMA50) to avoid counter-trend trades
# - Enters on breakout of Camarilla R4/S4 levels (strong continuation) or fade at R3/S3 (mean reversion in range)
# - Volume confirmation: current volume > 1.3x 20-period average to filter weak breakouts
# - Designed for 6h timeframe to capture medium-term swings with lower frequency (target: 12-30 trades/year)
# - Works in bull markets via R4 breakout continuation and in bear markets via S3 fade or R4 breakdown
# - Discrete position sizing (0.25) to minimize fee churn while maintaining meaningful exposure

name = "6h_12h_camarilla_breakout_volume_v1"
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
    if len(df_12h) < 20:
        return signals
    
    # Pre-compute 12h Camarilla levels (based on previous 12h bar's range)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r4 = close_12h + 1.5 * (high_12h - low_12h)
    camarilla_r3 = close_12h + 1.1 * (high_12h - low_12h)
    camarilla_s3 = close_12h - 1.1 * (high_12h - low_12h)
    camarilla_s4 = close_12h - 1.5 * (high_12h - low_12h)
    
    # Align Camarilla levels to 6h timeframe (wait for completed 12h bar)
    r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Pre-compute 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels from aligned arrays
        r4 = r4_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # 12h EMA trend bias
        ema_bias_long = close_price > ema_50_12h_aligned[i]
        ema_bias_short = close_price < ema_50_12h_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above R4 with volume and long trend bias (strong continuation)
        if close_price > r4 and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Long mean reversion: price drops to S3 with volume and long trend bias (fade in uptrend)
        elif close_price < s3 and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price breaks below S4 with volume and short trend bias (strong continuation)
        elif close_price < s4 and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Short mean reversion: price rises to R3 with volume and short trend bias (fade in downtrend)
        elif close_price > r3 and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions: return to mean (midpoint of R3-S3) or opposite Camarilla extreme
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to R3-S3 midpoint or reaches R4 (take profit)
            mid_point = (r3 + s3) / 2
            exit_long = close_price <= mid_point or close_price >= r4
        elif position == -1:
            # Exit short if price returns to R3-S3 midpoint or reaches S4 (take profit)
            mid_point = (r3 + s3) / 2
            exit_short = close_price >= mid_point or close_price <= s4
        
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