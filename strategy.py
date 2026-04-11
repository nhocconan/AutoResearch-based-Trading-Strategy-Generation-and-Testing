#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w ADX trend filter
# - Long: Price breaks above Donchian upper channel (20-period high) on 1d, volume > 2.0x 20-period 1w avg, 1w ADX(14) > 25
# - Short: Price breaks below Donchian lower channel (20-period low) on 1d, volume > 2.0x 20-period 1w avg, 1w ADX(14) > 25
# - Exit: Price returns to Donchian midpoint (10-period average of high/low) or ATR-based stop (2.5 ATR)
# - Uses discrete position sizing: ±0.28 to balance return and drawdown
# - Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# - Donchian breakouts capture strong momentum moves in both bull and bear markets
# - Volume confirmation ensures institutional participation
# - 1w ADX > 25 filters for strong trends to avoid whipsaw in ranging markets

name = "1d_1w_donchian_volume_adx_v2"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for volume and ADX filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr_1w = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    tr_14 = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 1w volume SMA(20) for volume confirmation
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Pre-compute Donchian channels on 1d timeframe
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Pre-compute ATR for stoploss (1d timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(donchian_period, n):  # Start after Donchian period warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        midpoint = donchian_mid[i]
        
        # Volume confirmation: current volume > 2.0x 20-period 1w average
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Trend filter: 1w ADX > 25 (indicates strong trend)
        adx_trend = adx_aligned[i] > 25
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above Donchian upper channel
        if close_price > upper_channel and vol_confirm and adx_trend:
            enter_long = True
        
        # Short breakdown: price breaks below Donchian lower channel
        if close_price < lower_channel and vol_confirm and adx_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to midpoint or ATR-based stop
            exit_long = (close_price <= midpoint) or (close_price <= entry_price - 2.5 * atr_14[i])
        elif position == -1:
            # Exit short if price returns to midpoint or ATR-based stop
            exit_short = (close_price >= midpoint) or (close_price >= entry_price + 2.5 * atr_14[i])
        
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