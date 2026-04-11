#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d volume spike and 1d ADX trend filter
# - Long: Williams %R(14) crosses above -80 (oversold reversal), volume > 1.8x 20-period avg, 1d ADX(14) > 25
# - Short: Williams %R(14) crosses below -20 (overbought reversal), volume > 1.8x 20-period avg, 1d ADX(14) > 25
# - Exit: Williams %R returns to -50 level or ATR-based stop (2.5 ATR)
# - Uses discrete position sizing: ±0.28 to balance return and drawdown
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Williams %R captures momentum reversals effectively in ranging and trending markets
# - Volume spike confirms institutional participation
# - 1d ADX > 25 ensures we only trade when there is sufficient trend strength to avoid whipsaw
# - Higher ATR multiplier (2.5) reduces premature stops in volatile markets

name = "4h_1d_williamsr_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for Williams %R, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute 1d Williams %R previous value for crossover detection
    williams_r_prev = np.roll(williams_r_aligned, 1)
    williams_r_prev[0] = williams_r_aligned[0]  # first value
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d_adx = df_1d['high'].values
    low_1d_adx = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    
    # True Range
    tr_1d = np.maximum(high_1d_adx - low_1d_adx, np.maximum(np.abs(high_1d_adx - np.roll(close_1d_adx, 1)), np.abs(low_1d_adx - np.roll(close_1d_adx, 1))))
    tr_1d[0] = high_1d_adx[0] - low_1d_adx[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d_adx - np.roll(high_1d_adx, 1)) > (np.roll(low_1d_adx, 1) - low_1d_adx), np.maximum(high_1d_adx - np.roll(high_1d_adx, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d_adx, 1) - low_1d_adx) > (high_1d_adx - np.roll(high_1d_adx, 1)), np.maximum(np.roll(low_1d_adx, 1) - low_1d_adx, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute ATR for stoploss (4h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(williams_r_prev[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Williams %R values
        wr_current = williams_r_aligned[i]
        wr_previous = williams_r_prev[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume_current > 1.8 * volume_sma_20_aligned[i]
        
        # Trend filter: 1d ADX > 25 (indicates sufficient trend strength)
        adx_trend = adx_aligned[i] > 25
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long reversal: Williams %R crosses above -80 (oversold bounce)
        if wr_previous <= -80 and wr_current > -80 and vol_confirm and adx_trend:
            enter_long = True
        
        # Short reversal: Williams %R crosses below -20 (overbought rejection)
        if wr_previous >= -20 and wr_current < -20 and vol_confirm and adx_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R returns to -50 or ATR-based stop
            exit_long = (wr_current >= -50) or (close_price <= entry_price - 2.5 * atr_14[i])
        elif position == -1:
            # Exit short if Williams %R returns to -50 or ATR-based stop
            exit_short = (wr_current <= -50) or (close_price >= entry_price + 2.5 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.28
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.28
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.28 if position == 1 else (-0.28 if position == -1 else 0.0)
    
    return signals