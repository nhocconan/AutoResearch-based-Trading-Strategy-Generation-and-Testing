#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with volume confirmation and 1w EMA trend filter
# - Long: price breaks above Camarilla H3 level, volume > 1.5x 20-period avg, price > 1w EMA(50)
# - Short: price breaks below Camarilla L3 level, volume > 1.5x 20-period avg, price < 1w EMA(50)
# - Exit: price returns to Camarilla pivot point (PP)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within fee drag limits
# - Camarilla levels derived from prior day's range work well in both trending and ranging markets
# - 1w EMA provides robust long-term trend filter effective in both bull and bear markets

name = "1d_1w_camarilla_ema_volume_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # PP = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # We'll use H3 (R3) and L3 (S3) for breakouts, PP for exit
    PP = (high_1d + low_1d + close_1d) / 3
    R3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    S3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 1d timeframe (same timeframe, no shift needed but using align for consistency)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        PP_level = PP_aligned[i]
        R3_level = R3_aligned[i]
        S3_level = S3_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Trend filter: price > 1w EMA(50) for long bias, price < 1w EMA(50) for short bias
        ema_bias_long = close_price > ema_50_1w_aligned[i]
        ema_bias_short = close_price < ema_50_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Camarilla R3, volume confirmation, long bias
        if close_price > R3_level and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below Camarilla S3, volume confirmation, short bias
        if close_price < S3_level and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point
            exit_long = close_price <= PP_level
        elif position == -1:
            # Exit short if price returns to pivot point
            exit_short = close_price >= PP_level
        
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