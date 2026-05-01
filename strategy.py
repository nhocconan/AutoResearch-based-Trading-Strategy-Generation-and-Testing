#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h trend filter and volume spike confirmation.
# Long when Williams %R(14) crosses above -80 (oversold) AND price > 12h EMA50 AND volume > 2x 20-bar average.
# Short when Williams %R(14) crosses below -20 (overbought) AND price < 12h EMA50 AND volume > 2x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Williams %R is effective in ranging markets which dominate 2025+.
# 12h EMA50 provides smooth trend alignment that adapts to both bull and bear regimes.
# Volume spike (2x average) ensures only high-conviction reversals are traded.
# Target: 80-120 total trades over 4 years (20-30/year) for 6h timeframe.

name = "6h_WilliamsR_12hEMA50_VolumeSpike_Reversal_v1"
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
    
    # Pre-compute Williams %R(14) - looks back 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    williams_r = -100 * (highest_high - close) / hl_range
    
    # Williams %R signals: cross above -80 (long), cross below -20 (short)
    williams_r_long_signal = (williams_r > -80) & (np.roll(williams_r, 1) <= -80)
    williams_r_short_signal = (williams_r < -20) & (np.roll(williams_r, 1) >= -20)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 calculation
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 12h trend: price above/below EMA50
    price_above_ema = close > ema_50_aligned
    price_below_ema = close < ema_50_aligned
    
    # Volume confirmation: current 6h volume > 2x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R, EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if vol_ma[i] <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = volume[i] > (vol_ma[i] * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 AND price > 12h EMA50 AND volume confirmation
            if (williams_r_long_signal[i] and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND price < 12h EMA50 AND volume confirmation
            elif (williams_r_short_signal[i] and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) OR price < 12h EMA50 (trend change)
            if (williams_r[i] < -50 and np.roll(williams_r, 1)[i] >= -50) or \
               not price_above_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) OR price > 12h EMA50 (trend change)
            if (williams_r[i] > -50 and np.roll(williams_r, 1)[i] <= -50) or \
               not price_below_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals