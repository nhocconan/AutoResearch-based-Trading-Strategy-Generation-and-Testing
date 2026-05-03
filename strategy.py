#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume confirmation.
# Camarilla pivot levels provide intraday support/resistance. Breakout above R3 or below S3
# indicates strong momentum. Filtered by 1d EMA34 trend (price > EMA34 for long, < EMA34 for short)
# and volume > 1.5x 20-period MA to avoid false signals. Works in bull via long breakouts
# and bear via short breakdowns when aligned with 1d trend. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 12h: based on previous 12h bar's OHLC
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # We need previous bar's OHLC, so we shift by 1
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + (camarilla_range * 1.1 / 4)
    s3 = prev_close - (camarilla_range * 1.1 / 4)
    
    # Volume regime: current 12h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime from 1d EMA34
        is_uptrend = close_val > ema_val
        is_downtrend = close_val < ema_val
        
        # Entry logic
        if position == 0:
            # Long: Price breaks above R3 AND uptrend AND volume spike
            if close_val > r3[i] and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND downtrend AND volume spike
            elif close_val < s3[i] and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price drops below S3 OR trend reverses (downtrend) OR volume drops
            if close_val < s3[i] or not is_uptrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price rises above R3 OR trend reverses (uptrend) OR volume drops
            if close_val > r3[i] or not is_downtrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals