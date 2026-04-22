#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4-hour volume-weighted average price (VWAP) reversion with 1-day trend filter
    # Uses VWAP deviation as mean reversion signal, filtered by daily EMA trend direction
    # Works in bull/bear via trend filter: only take long when above daily EMA (uptrend bias)
    # and short when below daily EMA (downtrend bias). VWAP deviation captures intraday mean reversion.
    # Targets ~20-30 trades/year to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate typical price and VWAP components
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    
    # Calculate VWAP (20-period on 4h)
    vwap_numerator = pd.Series(pv).rolling(window=20, min_periods=20).sum().values
    vwap_denominator = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    # VWAP deviation as percentage
    vwap_dev = (close - vwap) / vwap
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: above average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(vwap_dev[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price below VWAP (oversold) in uptrend (price > daily EMA50) with volume
            if vwap_dev[i] < -0.008 and close[i] > ema50_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price above VWAP (overbought) in downtrend (price < daily EMA50) with volume
            elif vwap_dev[i] > 0.008 and close[i] < ema50_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to VWAP or trend reversal
            if position == 1:
                if vwap_dev[i] > -0.002 or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if vwap_dev[i] < 0.002 or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Reversion_DailyEMA50_Trend_Volume_Session_v1"
timeframe = "4h"
leverage = 1.0