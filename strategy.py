#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 in bull trend (close > 1d EMA34) with volume > 2.0x 20-period MA.
# Short when price breaks below Camarilla S3 in bear trend (close < 1d EMA34) with volume spike.
# Uses discrete position sizing (0.30) to balance return and drawdown. 1d EMA34 provides strong trend filter.
# Camarilla pivot levels derived from 1d OHLC provide institutional reference points.
# Volume confirmation ensures participation. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets: trend filter ensures we only trade in direction of 1d momentum,
# while Camarilla breakouts provide clear entry/exit levels with proven efficacy on ETH/SOL.

name = "12h_Camarilla_R3S3_1dEMA34_Volume"
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
    
    # Get 1d data for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from 1d OHLC (using previous day's close for today's levels)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use the previous 1d bar's OHLC to calculate levels for current 12h bars
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use only completed 1d bar's data
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range
    s3 = prev_close - 1.1 * camarilla_range
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Camarilla breakout conditions
        breakout_up = close_val > r3_val  # break above R3
        breakout_down = close_val < s3_val  # break below S3
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_up and vol_spike:
                signals[i] = 0.30
                position = 1
            elif is_bear_trend and breakout_down and vol_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Camarilla breakout down OR trend reversal
            if breakout_down or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Camarilla breakout up OR trend reversal
            if breakout_up or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals