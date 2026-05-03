#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d EMA34 trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold) in bull trend (close > 1d EMA34) with volume > 2x 20-period MA.
# Short when Williams %R > -20 (overbought) in bear trend (close < 1d EMA34) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn while maintaining sufficient exposure.
# Williams %R captures short-term extremes that often reverse in ranging markets.
# 1d EMA34 provides intermediate-term trend filter to avoid counter-trend trades.
# Volume confirmation ensures reversals have institutional participation, reducing false signals.
# Target: 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "12h_WilliamsR_1dEMA34_Volume"
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
    
    # Get 1d data for EMA34 trend filter and Williams %R
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period) from 1d data
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - df_1d['close'].values) / (highest_high_14 - lowest_low_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        wr_value = williams_r_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Williams %R conditions
        is_oversold = wr_value < -80
        is_overbought = wr_value > -20
        
        # Entry logic
        if position == 0:
            if is_bull_trend and is_oversold and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and is_overbought and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -50 (reversal from oversold) OR trend reversal
            if wr_value > -50 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -50 (reversal from overbought) OR trend reversal
            if wr_value < -50 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals