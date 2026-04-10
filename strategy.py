#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# - Long when Alligator jaws (13-period SMMA) crosses above teeth (8-period SMMA) AND
#   price > lips (5-period SMMA) AND 1w close > 1w EMA21 (bullish weekly trend) AND
#   volume > 1.5x 20-period average
# - Short when jaws crosses below teeth AND price < lips AND 1w close < 1w EMA21 AND
#   volume > 1.5x 20-period average
# - Exit when jaws re-crosses teeth (trend weakening) or opposite signal occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 20-50 trades/year on 1d timeframe (80-200 total over 4 years)
# - Williams Alligator identifies trending vs ranging markets via SMMA relationships
# - Weekly EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false signals

name = "1d_1w_alligator_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d Williams Alligator (SMMA = smoothed moving average)
    # Jaws: 13-period SMMA of median price
    median_price = (high + low) / 2
    jaws = np.zeros_like(median_price)
    # First value: simple average
    jaws[12] = np.mean(median_price[0:13])
    # Subsequent values: SMMA formula
    for i in range(13, len(median_price)):
        jaws[i] = (jaws[i-1] * 12 + median_price[i]) / 13
    
    # Teeth: 8-period SMMA of median price
    teeth = np.zeros_like(median_price)
    teeth[7] = np.mean(median_price[0:8])
    for i in range(8, len(median_price)):
        teeth[i] = (teeth[i-1] * 7 + median_price[i]) / 8
    
    # Lips: 5-period SMMA of median price
    lips = np.zeros_like(median_price)
    lips[4] = np.mean(median_price[0:5])
    for i in range(5, len(median_price)):
        lips[i] = (lips[i-1] * 4 + median_price[i]) / 5
    
    # Pre-compute 1w EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Align HTF indicators to 1d timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1w, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: jaws > teeth AND price > lips AND weekly bullish AND volume spike
            if (jaws_aligned[i] > teeth_aligned[i] and 
                close[i] > lips_aligned[i] and 
                close[i] > ema_21_1w_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: jaws < teeth AND price < lips AND weekly bearish AND volume spike
            elif (jaws_aligned[i] < teeth_aligned[i] and 
                  close[i] < lips_aligned[i] and 
                  close[i] < ema_21_1w_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when jaws re-crosses teeth (trend weakening) or opposite Alligator signal
            exit_long = (position == 1 and jaws_aligned[i] <= teeth_aligned[i])
            exit_short = (position == -1 and jaws_aligned[i] >= teeth_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals