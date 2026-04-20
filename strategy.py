#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d CCI + Volume Spike + 1w Trend Filter
# Uses CCI(20) for mean reversion at extremes, volume confirmation for breakout strength,
# and weekly EMA filter to avoid counter-trend trades. Works in bull/bear by
# fading extremes in ranging markets and following trend in trending markets.
# Target: 15-25 trades/year, low frequency to minimize fee drag.

name = "1d_CCI_VolumeSpike_WeekTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Daily CCI(20) ===
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    tp_series = pd.Series(typical_price.values)
    sma_tp = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp_series - sma_tp) / (0.015 * mad.replace(0, np.nan))
    cci_values = cci.values
    
    # === Volume Spike Detection ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / np.where(vol_ma20.values > 0, vol_ma20.values, np.nan)
    
    # === Weekly EMA Trend Filter ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        cci_val = cci_values[i]
        vol_ratio_val = vol_ratio[i]
        ema50_1w_val = ema50_1w_aligned[i]
        ema200_1w_val = ema200_1w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(cci_val) or np.isnan(vol_ratio_val) or 
            np.isnan(ema50_1w_val) or np.isnan(ema200_1w_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CCI < -100 (oversold) + volume spike + weekly uptrend
            if cci_val < -100 and vol_ratio_val > 2.0 and ema50_1w_val > ema200_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: CCI > 100 (overbought) + volume spike + weekly downtrend
            elif cci_val > 100 and vol_ratio_val > 2.0 and ema50_1w_val < ema200_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CCI returns above -50 OR trend breaks down
            if cci_val > -50 or ema50_1w_val < ema200_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI returns below 50 OR trend breaks up
            if cci_val < 50 or ema50_1w_val > ema200_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals