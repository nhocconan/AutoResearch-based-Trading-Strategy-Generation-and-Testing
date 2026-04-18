# 1d_HeikinAshi_SuperTrend_1wVolatility  
# Uses Heikin-Ashi candles to reduce noise and SuperTrend for trend identification on daily timeframe.  
# 1w volatility filter (ATR percentile) ensures we only trade when volatility is elevated,  
# which helps capture trends in both bull and bear markets while avoiding choppy periods.  
# Heikin-Ashi smooths price action, making trend changes clearer and reducing false signals.  
# SuperTrend provides clear entry/exit levels with ATR-based trailing stops.  
# Target: 15-40 trades per year to stay within optimal frequency for 1d timeframe.  

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_HeikinAshi_SuperTrend_1wVolatility"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate Heikin-Ashi candles
    ha_close = (prices['open'] + prices['high'] + prices['low'] + prices['close']) / 4
    ha_open = np.zeros(n)
    ha_open[0] = prices['open'][0]
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum(prices['high'], np.maximum(ha_open, ha_close))
    ha_low = np.minimum(prices['low'], np.minimum(ha_open, ha_close))
    
    # Get 1d data for SuperTrend calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for SuperTrend (10-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    atr[9] = np.mean(tr[:10])
    for i in range(10, len(tr)):
        atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # SuperTrend calculation (10, 3.0)
    factor = 3.0
    upper_band = (high_1d + low_1d) / 2 + factor * atr
    lower_band = (high_1d + low_1d) / 2 - factor * atr
    
    super_trend = np.zeros_like(close_1d)
    super_trend[:] = np.nan
    uptrend = np.ones_like(close_1d, dtype=bool)
    
    for i in range(10, len(close_1d)):
        if close_1d[i] > upper_band[i-1]:
            uptrend[i] = True
        elif close_1d[i] < lower_band[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if not uptrend[i] and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if uptrend[i]:
            super_trend[i] = lower_band[i]
        else:
            super_trend[i] = upper_band[i]
    
    # Align SuperTrend and trend direction to 1d timeframe
    super_trend_aligned = align_htf_to_ltf(prices, df_1d, super_trend)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    
    # Get 1w data for volatility filter (ATR percentile)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR for volatility filter
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr1_w[0] = high_1w[0] - low_1w[0]
    tr2_w[0] = tr1_w[0]
    tr3_w[0] = tr1_w[0]
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    atr_w = np.zeros_like(tr_w)
    atr_w[19] = np.mean(tr_w[:20])
    for i in range(20, len(tr_w)):
        atr_w[i] = (atr_w[i-1] * 19 + tr_w[i]) / 20
    
    # Calculate ATR percentile rank (50-period lookback)
    atr_percentile = np.zeros_like(atr_w)
    for i in range(50, len(atr_w)):
        lookback = atr_w[max(0, i-49):i+1]
        rank = np.sum(lookback <= atr_w[i]) / len(lookback) * 100
        atr_percentile[i] = rank
    
    # Align volatility filter to 1d timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(super_trend_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or
            np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        ha_close_val = ha_close.iloc[i]
        ha_open_val = ha_open[i]
        ha_high_val = ha_high[i]
        ha_low_val = ha_low[i]
        
        super_trend_val = super_trend_aligned[i]
        uptrend_val = bool(uptrend_aligned[i])
        vol_filter_val = atr_percentile_aligned[i]
        
        # Volatility filter: only trade when volatility is above 40th percentile
        volatility_filter = vol_filter_val > 40
        
        if position == 0:
            # Long: Heikin-Ashi bullish (close > open) AND uptrend AND volatility filter
            if ha_close_val > ha_open_val and uptrend_val and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: Heikin-Ashi bearish (close < open) AND downtrend AND volatility filter
            elif ha_close_val < ha_open_val and not uptrend_val and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Heikin-Ashi turns bearish OR trend reverses OR volatility drops
            if ha_close_val < ha_open_val or not uptrend_val or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Heikin-Ashi turns bullish OR trend reverses OR volatility drops
            if ha_close_val > ha_open_val or uptrend_val or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals