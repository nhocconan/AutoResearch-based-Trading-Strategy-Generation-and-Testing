#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Volume-Weighted Average Price (VWAP) Deviation with Daily Trend Filter
# Uses VWAP deviation for mean reversion in ranging markets, aligned with daily trend.
# In bull/bear markets: follow daily trend when price deviates from VWAP with volume confirmation.
# Designed for 20-50 trades/year to avoid fee drag. Works in all regimes.
name = "4h_VWAP_Deviation_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(df_daily['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Typical price for VWAP calculation
    typical_price = (high + low + close) / 3.0
    
    # VWAP calculation (typical price * volume) / cumulative volume
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    vwap_4h = align_htf_to_ltf(prices, pd.DataFrame({'vwap': vwap}, index=prices.index[:len(vwap)]), vwap)
    
    # VWAP deviation as percentage
    vwap_dev = (close - vwap_4h) / vwap_4h * 100
    
    # 20-period standard deviation of VWAP deviation for volatility normalization
    vwap_dev_ma = pd.Series(vwap_dev).rolling(window=20, min_periods=20).mean().values
    vwap_dev_std = pd.Series(vwap_dev).rolling(window=20, min_periods=20).std().values
    
    # Z-score of VWAP deviation
    vwap_zscore = np.divide((vwap_dev - vwap_dev_ma), vwap_dev_std, out=np.zeros_like(vwap_dev), where=vwap_dev_std!=0)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_4h[i]) or np.isnan(vwap_zscore[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: VWAP deviation < -1.5 (oversold) with daily uptrend and volume spike
            if vwap_zscore[i] < -1.5 and close[i] > ema34_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: VWAP deviation > 1.5 (overbought) with daily downtrend and volume spike
            elif vwap_zscore[i] > 1.5 and close[i] < ema34_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VWAP deviation > -0.5 (mean reversion) OR daily trend turns down
            if vwap_zscore[i] > -0.5 or close[i] < ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VWAP deviation < 0.5 (mean reversion) OR daily trend turns up
            if vwap_zscore[i] < 0.5 or close[i] > ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals