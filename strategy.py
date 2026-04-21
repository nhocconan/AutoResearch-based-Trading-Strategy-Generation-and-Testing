#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R(14) + 1d EMA(34) Trend + Volume Confirmation
# Long when Williams %R < -80 (oversold) and price > 1d EMA(34) and 1d volume > 1.5x 20-period average
# Short when Williams %R > -20 (overbought) and price < 1d EMA(34) and 1d volume > 1.5x 20-period average
# Exit when Williams %R crosses -50 (mean reversion) or opposite signal triggers
# Williams %R identifies overextended moves in 12h timeframe
# 1d EMA(34) ensures we trade with the higher timeframe trend
# Volume confirmation avoids false reversals
# Target: 15-30 trades/year by requiring strict overextension + trend alignment + volume

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(34) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Williams %R(14) on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # Neutral when no range
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = prices['close'].iloc[i]
        williams = williams_r[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        # Need to get current 1d volume for this 12h bar
        # Approximate: 12h bar is half of 1d, so use current 1d volume
        volume_confirm = df_1d['volume'].iloc[i // 2] > 1.5 * vol_ma if i >= 2 else df_1d['volume'].iloc[0] > 1.5 * vol_ma
        
        if position == 0:
            if volume_confirm:
                # Long: oversold + above trend EMA
                if williams < -80 and price > ema_trend:
                    signals[i] = 0.25
                    position = 1
                # Short: overbought + below trend EMA
                elif williams > -20 and price < ema_trend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R crosses above -50 (overbought territory) or price breaks below EMA
                if williams > -50 or price < ema_trend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R crosses below -50 (oversold territory) or price breaks above EMA
                if williams < -50 or price > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR14_1dEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0