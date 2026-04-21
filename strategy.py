#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels: long when %R < -80 (oversold) and price > 1d EMA50,
# short when %R > -20 (overbought) and price < 1d EMA50.
# Requires volume > 1.5x 20-period average to confirm momentum.
# Target: 20-50 trades/year by combining mean reversion with trend filter and volume confirmation.
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R (14-period)
    high_14 = pd.Series(prices['high'].values).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(prices['low'].values).rolling(window=14, min_periods=14).min().values
    close = prices['close'].values
    williams_r = -100 * (high_14 - close) / (high_14 - low_14)
    # Handle division by zero when high == low
    williams_r = np.where((high_14 - low_14) == 0, -50, williams_r)
    
    # Calculate 20-period volume average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Williams %R levels
        wr = williams_r[i]
        oversold = wr < -80
        overbought = wr > -20
        
        # Trend filter: price vs 1d EMA50
        bull_trend = price > ema50_1d_aligned[i]
        bear_trend = price < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Enter long when oversold in bullish trend with volume confirmation
            if oversold and bull_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short when overbought in bearish trend with volume confirmation
            elif overbought and bear_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when Williams %R reverts to neutral territory (-50)
            exit_signal = False
            
            if position == 1:
                # Exit long when %R rises above -50 (overbought territory)
                if wr > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when %R falls below -50 (oversold territory)
                if wr < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_Trend_Volume"
timeframe = "4h"
leverage = 1.0