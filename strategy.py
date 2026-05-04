#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w trend filter and volume confirmation
# Long when price breaks above Camarilla R3 level with 1w bullish trend (close > EMA34) and volume > 1.3x 20-period volume EMA
# Short when price breaks below Camarilla S3 level with 1w bearish trend (close < EMA34) and volume > 1.3x 20-period volume EMA
# Uses Camarilla pivot levels from prior 1d for structure, 1w EMA34 for major trend filter to reduce whipsaw, and moderate volume confirmation.
# Targeting 7-25 trades/year on 1d. Works in bull markets via longs in bullish 1w trend regime and bear markets via shorts in bearish 1w trend regime.
# Camarilla levels provide tighter structure than Donchian, reducing false breakouts and overtrading.

name = "1d_Camarilla_R3S3_1wTrend_VolumeSpike"
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
    trend_bullish_1w = close_1w > ema_34_1w
    trend_bearish_1w = close_1w < ema_34_1w
    
    # Align 1w trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Calculate Camarilla levels from prior 1d bar (HLC of previous bar)
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2 where C,H,L are prior bar close,high,low
    # We need prior bar values, so shift by 1
    prior_close = np.roll(close, 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    # Set first value to NaN since no prior bar
    prior_close[0] = np.nan
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 2
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 2
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.3)  # Volume at least 1.3x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1w bullish trend AND volume spike
            if (close[i] > camarilla_r3[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1w bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1w bearish trend AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1w bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below prior bar close OR 1w trend turns bearish
            if (close[i] < prior_close[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above prior bar close OR 1w trend turns bullish
            if (close[i] > prior_close[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals