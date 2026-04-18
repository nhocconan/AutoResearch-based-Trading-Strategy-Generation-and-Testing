#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtrader.indicators import cci

# Hypothesis: 4h CCI(20) mean reversion with 1d trend filter and volume confirmation.
# CCI identifies overbought/oversold conditions. Enters when CCI crosses back into normal range
# (-100 to 100) after extreme readings, aligned with higher timeframe trend and volume spike.
# Designed for 15-30 trades/year to avoid fee drag while capturing mean reversion in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate CCI(20) on 4h data
    typical_price = (high + low + close) / 3.0
    ma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_values = (typical_price - ma_tp) / (0.015 * mad)
    cci_values = np.where(mad == 0, 0, cci_values)  # avoid division by zero
    
    # Get 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # CCI and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cci_values[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: price above 1d EMA50 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: CCI crosses above -100 from oversold, with volume and uptrend
            if (cci_values[i] > -100 and cci_values[i-1] <= -100 and
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: CCI crosses below 100 from overbought, with volume and downtrend
            elif (cci_values[i] < 100 and cci_values[i-1] >= 100 and
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: CCI crosses above 100 (overbought) or crosses below zero
            if cci_values[i] >= 100 or (cci_values[i] < 0 and cci_values[i-1] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI crosses below -100 (oversold) or crosses above zero
            if cci_values[i] <= -100 or (cci_values[i] > 0 and cci_values[i-1] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_CCI20_1dEMA50_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0