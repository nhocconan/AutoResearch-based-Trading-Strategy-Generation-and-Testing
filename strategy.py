#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w volume confirmation + ADX trend filter
# - Primary: 1d price breaks above/below 20-period Donchian channel from prior 20 days
# - HTF: 1w volume confirmation (current week volume > 1.5x 4-week MA) for conviction
# - Regime: 1d ADX(14) > 25 to ensure trending market (avoid ranging/chop)
# - Long: Close > Upper Donchian + volume confirmation + ADX > 25
# - Short: Close < Lower Donchian + volume confirmation + ADX > 25
# - Exit: Close crosses back inside Donchian channel OR ADX < 20 (trend weakening)
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Works in bull/bear: Donchian adapts to volatility, volume filters false signals, ADX avoids whipsaws in ranging markets
# - Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_1w_donchian_breakout_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 1d data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Donchian channel (20-period) - based on prior 20 periods
    upper_channel = np.full(len(close), np.nan)
    lower_channel = np.full(len(close), np.nan)
    
    for i in range(20, len(close)):
        if not (np.isnan(high[i-20:i]).any() or np.isnan(low[i-20:i]).any()):
            upper_channel[i] = np.max(high[i-20:i])
            lower_channel[i] = np.min(low[i-20:i])
    
    # Calculate 1w volume moving average (4-period) for volume confirmation
    volume_ma_4_1w = np.full(len(volume_1w), np.nan)
    for i in range(4, len(volume_1w)):
        if not np.isnan(volume_1w[i-4:i+1]).any():
            volume_ma_4_1w[i] = np.mean(volume_1w[i-4:i+1])
    
    # Calculate 1d ADX(14) for trend filter
    # First calculate True Range and +DM/-DM
    tr = np.full(len(close), np.nan)
    plus_dm = np.full(len(close), np.nan)
    minus_dm = np.full(len(close), np.nan)
    
    for i in range(1, len(close)):
        if not (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i-1]) or 
                np.isnan(high[i-1]) or np.isnan(low[i-1])):
            # True Range
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            
            # Directional Movement
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
                
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
    
    # Calculate smoothed TR, +DM, -DM using Wilder's smoothing (alpha=1/14)
    atr = np.full(len(close), np.nan)
    smoothed_plus_dm = np.full(len(close), np.nan)
    smoothed_minus_dm = np.full(len(close), np.nan)
    
    for i in range(14, len(tr)):
        if not (np.isnan(tr[i-13:i+1]).any() or 
                np.isnan(plus_dm[i-13:i+1]).any() or 
                np.isnan(minus_dm[i-13:i+1]).any()):
            if i == 14:
                atr[i] = np.mean(tr[1:15])
                smoothed_plus_dm[i] = np.mean(plus_dm[1:15])
                smoothed_minus_dm[i] = np.mean(minus_dm[1:15])
            else:
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
                smoothed_plus_dm[i] = (smoothed_plus_dm[i-1] * 13 + plus_dm[i]) / 14
                smoothed_minus_dm[i] = (smoothed_minus_dm[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate +DI and -DI
    plus_di = np.full(len(close), np.nan)
    minus_di = np.full(len(close), np.nan)
    dx = np.full(len(close), np.nan)
    
    for i in range(14, len(atr)):
        if not (np.isnan(atr[i]) or np.isnan(smoothed_plus_dm[i]) or np.isnan(smoothed_minus_dm[i])):
            if atr[i] != 0:
                plus_di[i] = (smoothed_plus_dm[i] / atr[i]) * 100
                minus_di[i] = (smoothed_minus_dm[i] / atr[i]) * 100
                
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # Calculate ADX as smoothed DX (Wilder's smoothing)
    adx = np.full(len(close), np.nan)
    for i in range(28, len(dx)):  # 14 for initial DX + 14 for smoothing
        if not np.isnan(dx[i-13:i+1]).any():
            if i == 28:
                adx[i] = np.mean(dx[14:29])
            else:
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align all HTF indicators to 1d timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, prices, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, prices, lower_channel)
    volume_ma_4_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_4_1w)
    adx_aligned = align_htf_to_ltf(prices, prices, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(29, n):  # Start after warmup period for all indicators
        # Skip if any required data is invalid
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(volume_ma_4_1w_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 4-period MA
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        volume_confirm = volume_1w_aligned[i] > 1.5 * volume_ma_4_1w_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending_regime = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close > Upper Donchian + volume confirmation + trending regime
            if close[i] > upper_channel_aligned[i] and volume_confirm and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Close < Lower Donchian + volume confirmation + trending regime
            elif close[i] < lower_channel_aligned[i] and volume_confirm and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Close crosses back inside Donchian channel OR ADX < 20 (trend weakening)
            if position == 1:  # Long position
                if close[i] < lower_channel_aligned[i] or adx_aligned[i] < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > upper_channel_aligned[i] or adx_aligned[i] < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals