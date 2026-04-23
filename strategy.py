#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme Reversal with 1d EMA34 trend filter and volume spike confirmation.
- Williams %R(14) < -80 = oversold (long signal), > -20 = overbought (short signal)
- Long: Williams %R crosses above -80 + price > 1d EMA34 + volume > 1.5x 20-period avg volume
- Short: Williams %R crosses below -20 + price < 1d EMA34 + volume > 1.5x 20-period avg volume
- Exit: ATR-based trailing stop (2.5x ATR from extreme) OR Williams %R crosses opposite extreme (-50 for exit)
- Uses 1d EMA34 as trend filter to avoid counter-trend trades in strong trends
- Volume confirmation reduces false reversals in low-momentum environments
- Williams %R is effective in ranging/bear markets (2025 test period) for mean reversion
- Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Williams %R and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R(14) on 1d data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate EMA34 on 1d data
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 34)  # Need 20 for volume MA, 14 for ATR/Williams, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Williams %R conditions
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1] if i > 0 else wr
        
        # Bullish reversal: crosses above -80 from below
        bullish_reversal = (wr_prev <= -80) and (wr > -80)
        # Bearish reversal: crosses below -20 from above
        bearish_reversal = (wr_prev >= -20) and (wr < -20)
        # Exit conditions: cross -50 (middle level)
        exit_long = (wr_prev > -50) and (wr <= -50)
        exit_short = (wr_prev < -50) and (wr >= -50)
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R bullish reversal + price > 1d EMA34 + volume spike
            if bullish_reversal and close[i] > ema_34_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Williams %R bearish reversal + price < 1d EMA34 + volume spike
            elif bearish_reversal and close[i] < ema_34_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from long extreme (trailing stop)
            # 2. Williams %R crosses below -50 (momentum weakening)
            trailing_stop_long = close[i] < long_extreme - 2.5 * atr[i]
            momentum_exit = exit_long
            
            if trailing_stop_long or momentum_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from short extreme (trailing stop)
            # 2. Williams %R crosses above -50 (momentum weakening)
            trailing_stop_short = close[i] > short_extreme + 2.5 * atr[i]
            momentum_exit = exit_short
            
            if trailing_stop_short or momentum_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dEMA34_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0