#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extremes with 1d EMA34 trend filter and volume confirmation.
# Enter long when Williams %R < -80 (oversold), price > 1d EMA34 (uptrend), and volume > 1.5x 20-bar average.
# Enter short when Williams %R > -20 (overbought), price < 1d EMA34 (downtrend), and volume > 1.5x 20-bar average.
# Exit when Williams %R crosses -50 (mean reversion) or price crosses 1d EMA34.
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Target: 80-120 total trades over 4 years (20-30/year) to avoid excessive fee churn.
# Williams %R captures momentum extremes; 1d EMA34 filters for higher timeframe trend alignment;
# Volume confirmation ensures institutional participation in reversals.

name = "6h_WilliamsR_Extremes_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Williams %R conditions
        wr_oversold = williams_r[i] < -80
        wr_overbought = williams_r[i] > -20
        wr_exit = williams_r[i] > -50  # Exit when crosses above -50 (mean reversion)
        
        # 1d EMA34 trend
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]
        
        # Handle entries and exits
        if wr_oversold and price_above_ema and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif wr_overbought and price_below_ema and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and (wr_exit or price_below_ema):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (wr_exit or price_above_ema):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals