#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume surge and 1w EMA50 trend filter
    # Camarilla levels provide precise intraday support/resistance with high win rate
    # Breakout of R3/S3 indicates strong momentum; volume surge confirms institutional interest
    # 1w EMA50 filter ensures trading with the dominant weekly trend, avoiding counter-trend whipsaws
    # This combination works in both bull (buy R3 breaks in uptrend) and bear (sell S3 breaks in downtrend)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    df_1d = get_htf_data(prices, '1d')
    camarilla_R3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    camarilla_S3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3.values)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3.values)
    
    # 1d volume surge (20-period average)
    vol_ma20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_surge = df_1d['volume'] > 2.0 * vol_ma20  # Require 2x volume for confirmation
    vol_surge_aligned = align_htf_to_ltf(prices, df_1d, vol_surge)
    
    # 1w EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_surge_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R3 + volume surge + above weekly EMA50 (uptrend)
            if close[i] > camarilla_R3_aligned[i] and vol_surge_aligned[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + volume surge + below weekly EMA50 (downtrend)
            elif close[i] < camarilla_S3_aligned[i] and vol_surge_aligned[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to Camarilla center (PPT) or trend reversal vs weekly EMA50
            camarilla_PPT = (df_1d['close'] + df_1d['high'] + df_1d['low']) / 3
            camarilla_PPT_aligned = align_htf_to_ltf(prices, df_1d, camarilla_PPT.values)
            
            if position == 1:
                if close[i] < camarilla_PPT_aligned[i] or close[i] < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > camarilla_PPT_aligned[i] or close[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSurge_1wEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0