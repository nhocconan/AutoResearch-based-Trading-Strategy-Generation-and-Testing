#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extremes with 1d EMA34 trend filter and volume confirmation.
# Enter long when Williams %R(14) < -80 (oversold) AND price > 1d EMA34 (bullish bias) AND volume > 1.5x 20-bar average.
# Enter short when Williams %R(14) > -20 (overbought) AND price < 1d EMA34 (bearish bias) AND volume > 1.5x 20-bar average.
# Exit when Williams %R returns to neutral zone (-50) or crosses opposite extreme.
# Williams %R captures mean reversion in 6h swings; 1d EMA34 filters for higher timeframe trend alignment;
# Volume confirmation ensures institutional participation in reversals.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag.

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
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
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
        wr = williams_r[i]
        wr_oversold = wr < -80
        wr_overbought = wr > -20
        wr_neutral = wr >= -50 and wr <= -50  # Exit at -50 level
        
        # 1d EMA34 trend filter
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]
        
        # Exit conditions: Williams %R returns to neutral (-50) or crosses opposite extreme
        exit_long = wr >= -50  # Exit long when WR returns to neutral
        exit_short = wr <= -50  # Exit short when WR returns to neutral
        
        # Handle entries and exits
        if wr_oversold and price_above_ema and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif wr_overbought and price_below_ema and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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