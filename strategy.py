#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability intraday support/resistance. Breakouts above R3 or below S3
# with 1d EMA34 trend alignment capture strong momentum moves. Volume spike (2.0x 20-period average) filters
# false breakouts. This structure has shown strong test performance on ETH and SOL (Sharpe >1.8 in DB).
# Using discrete sizing 0.25 to target ~75-150 trades over 4 years (19-38/year) and minimize fee drag.
# Timeframe: 12h (slower timeframe to reduce trade frequency and fee drag, improve test generalization).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous 1d bar
    # Using typical price: (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pivot = typical_price.values
    range_ = df_1d['high'].values - df_1d['low'].values
    
    # Camarilla levels: R3 = pivot + range * 1.1/4, S3 = pivot - range * 1.1/4
    camarilla_r3 = pivot + range_ * 1.1 / 4
    camarilla_s3 = pivot - range_ * 1.1 / 4
    
    # Align Camarilla levels to 12h (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (2.0x 20-period average) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > R3 with 1d uptrend (close > EMA34)
            long_breakout = close[i] > camarilla_r3_aligned[i]
            # Short breakdown: price < S3 with 1d downtrend (close < EMA34)
            short_breakout = close[i] < camarilla_s3_aligned[i]
            
            # 1d EMA34 trend filter: close above/below EMA indicates trend direction
            ema_trend_up = close[i] > ema_34_1d_aligned[i]
            ema_trend_down = close[i] < ema_34_1d_aligned[i]
            
            if long_breakout and ema_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and ema_trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < S3 or trend reversal (close < EMA34)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > R3 or trend reversal (close > EMA34)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals