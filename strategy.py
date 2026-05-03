#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level in bull trend (close > 1d EMA34) with volume > 2.0x 20-period MA.
# Short when price breaks below Camarilla S3 level in bear trend (close < 1d EMA34) with volume spike.
# Uses discrete position sizing (0.30) to balance return and drawdown. Camarilla levels provide high-probability reversal/breakout structure.
# Volume confirmation ensures institutional participation. 1d trend filter reduces whipsaw vs shorter MAs.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "4h_Camarilla_R3S3_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align EMA to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels using previous day's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # But we need daily OHLC - use get_htf_data for 1d
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    camarilla_R3 = np.zeros(len(df_1d_ohlc))
    camarilla_S3 = np.zeros(len(df_1d_ohlc))
    
    for i in range(len(df_1d_ohlc)):
        daily_high = df_1d_ohlc['high'].iloc[i]
        daily_low = df_1d_ohlc['low'].iloc[i]
        daily_close = df_1d_ohlc['close'].iloc[i]
        range_val = daily_high - daily_low
        
        camarilla_R3[i] = daily_close + 1.125 * range_val
        camarilla_S3[i] = daily_close - 1.125 * range_val
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_S3)
    
    # Volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_level = camarilla_R3_aligned[i]
        s3_level = camarilla_S3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Camarilla breakout conditions
        breakout_R3 = close_val > r3_level
        breakout_S3 = close_val < s3_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_R3 and vol_spike:
                signals[i] = 0.30
                position = 1
            elif is_bear_trend and breakout_S3 and vol_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR trend reversal
            if close_val < s3_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above R3 OR trend reversal
            if close_val > r3_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals