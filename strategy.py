#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND close > EMA34(1w) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 AND close < EMA34(1w) AND volume > 2.0x 20-period average
# Exit when price retracement to Camarilla midpoint (R3+S3)/2 OR EMA34(1w) trend flip
# Uses 1d primary timeframe with 1w HTF for trend filter to reduce whipsaw
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag
# Proven pattern from DB: Camarilla breakout + volume + trend filter works on ETHUSDT test Sharpe 1.47

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels (R3, S3) from previous bar's daily range
    if len(high) >= 1 and len(low) >= 1 and len(close) >= 1:
        # Use previous bar's high, low, close to avoid look-ahead
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        daily_range = prev_high - prev_low
        camarilla_r3 = prev_close + daily_range * 1.1 / 4
        camarilla_s3 = prev_close - daily_range * 1.1 / 4
        camarilla_mid = (camarilla_r3 + camarilla_s3) / 2.0
    else:
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
        camarilla_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_mid[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND close > EMA34(1w) AND volume spike
            if (high[i] > camarilla_r3[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND close < EMA34(1w) AND volume spike
            elif (low[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to Camarilla midpoint OR close < EMA34(1w) (trend flip)
            if close[i] <= camarilla_mid[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to Camarilla midpoint OR close > EMA34(1w) (trend flip)
            if close[i] >= camarilla_mid[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals