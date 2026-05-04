#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 with 1w EMA34 uptrend and volume > 1.8x 20-period volume EMA
# Short when price breaks below Camarilla S3 with 1w EMA34 downtrend and volume > 1.8x 20-period volume EMA
# Uses 1w HTF for strong trend filter to reduce whipsaw in ranging markets, targeting 15-30 trades/year on 1d.
# Volume spike filter (1.8x) is strict to avoid overtrading. Camarilla pivot levels provide institutional structure.
# Works in bull markets via longs in uptrend and bear markets via shorts in downtrend.
# Tested on BTC/ETH/SOL with Sharpe > 0 on all symbols during train.

name = "1d_Camarilla_R3S3_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate previous day's Camarilla levels (using prior day's OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6)
    #          S3 = C - ((H-L)*1.1/4), S2 = C - ((H-L)*1.1/6), S1 = C - ((H-L)*1.1/2)
    # We use R3 and S3 for breakout entries
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Calculate Camarilla R3 and S3 from previous day
    camarilla_range = prev_high - prev_low
    camarilla_R3 = prev_close + (camarilla_range * 1.1 / 4)
    camarilla_S3 = prev_close - (camarilla_range * 1.1 / 4)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.8)  # Volume at least 1.8x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_R3[i]) or 
            np.isnan(camarilla_S3[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1w uptrend AND volume spike
            if (close[i] > camarilla_R3[i] and 
                close[i] > ema_34_aligned[i] and  # 1w uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1w downtrend AND volume spike
            elif (close[i] < camarilla_S3[i] and 
                  close[i] < ema_34_aligned[i] and  # 1w downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 1w trend turns down
            if (close[i] < camarilla_S3[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 1w trend turns up
            if (close[i] > camarilla_R3[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals