#!/usr/bin/env python3
"""
Hypothesis: 4h RSI mean reversion with Bollinger Bands and volume confirmation.
In ranging markets, price tends to revert to the mean after reaching extreme RSI levels.
Bollinger Bands provide dynamic support/resistance, and volume spikes confirm reversal signals.
Designed for low trade frequency (<50/year) to minimize fee drag in bear markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate RSI (14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands (20, 2.0) on 4h
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Calculate 1d RSI for trend filter
    delta_daily = np.diff(df_daily['close'].values, prepend=df_daily['close'].values[0])
    gain_daily = np.where(delta_daily > 0, delta_daily, 0)
    loss_daily = np.where(delta_daily < 0, -delta_daily, 0)
    avg_gain_daily = pd.Series(gain_daily).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_daily = pd.Series(loss_daily).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_daily = avg_gain_daily / (avg_loss_daily + 1e-10)
    rsi_daily = 100 - (100 / (1 + rs_daily))
    rsi_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi_daily)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(rsi_daily_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30), price near BB lower band, bullish daily RSI (>50), volume spike
            if (rsi[i] < 30 and 
                close[i] <= bb_lower[i] * 1.02 and  # Within 2% of lower BB
                rsi_daily_aligned[i] > 50 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70), price near BB upper band, bearish daily RSI (<50), volume spike
            elif (rsi[i] > 70 and 
                  close[i] >= bb_upper[i] * 0.98 and  # Within 2% of upper BB
                  rsi_daily_aligned[i] < 50 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral (40-60) or price reaches opposite BB
            if position == 1:
                if rsi[i] > 50 or close[i] >= bb_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi[i] < 50 or close[i] <= bb_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_RSI_MeanReversion_BB_Volume"
timeframe = "4h"
leverage = 1.0
#%%