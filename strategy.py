#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation
# Targets 20-50 trades/year (80-200 total over 4 years) on 4h timeframe
# Camarilla pivot levels provide high-probability intraday support/resistance
# 1d EMA34 filters for higher timeframe trend alignment to avoid counter-trend trades
# Volume spike (current volume > 2.0x 20-period average) confirms institutional participation
# Works in bull markets (breakouts with trend + volume) and bear markets (breakdowns with trend + volume)
# Discrete position sizing (0.30) balances return potential with drawdown control
# Designed to avoid overtrading by requiring confluence of price structure, trend, and volume

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    
    # For 4h timeframe, we need to calculate pivots based on daily OHLC
    # We'll use the previous day's OHLC to calculate today's Camarilla levels
    # But since we're on 4h chart, we need to align the daily pivot levels
    
    # Get daily OHLC data
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # Camarilla R1 = close + ((high - low) * 1.1 / 12)
    # Camarilla S1 = close - ((high - low) * 1.1 / 12)
    high_1d = df_1d_ohlc['high'].values
    low_1d = df_1d_ohlc['low'].values
    close_1d = df_1d_ohlc['close'].values
    
    camarilla_r1 = close_1d + ((high_1d - low_1d) * 1.1 / 12.0)
    camarilla_s1 = close_1d - ((high_1d - low_1d) * 1.1 / 12.0)
    
    # Align the daily Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_s1)
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Price breaks above Camarilla R1 with volume confirmation
                if close[i] > camarilla_r1_aligned[i-1] and volume_confirmation[i]:
                    signals[i] = 0.30
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Price breaks below Camarilla S1 with volume confirmation
                if close[i] < camarilla_s1_aligned[i-1] and volume_confirmation[i]:
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around 1d EMA34
        
        elif position == 1:  # Long position
            # Exit: Price closes below Camarilla S1 or trend reverses
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price closes above Camarilla R1 or trend reverses
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals