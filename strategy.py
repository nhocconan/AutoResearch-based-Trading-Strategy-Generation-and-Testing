#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR volatility filter + volume confirmation
# - Primary: 4h Donchian channel breakout (20-period) for trend capture
# - HTF: 1d ATR(14) filter to avoid breakouts in low volatility (ATR > 0.8 * 20-period MA of ATR)
# - Volume: 4h volume > 1.2 * 20-period MA of volume for confirmation
# - Long: Price breaks above upper Donchian + volatility filter + volume confirmation
# - Short: Price breaks below lower Donchian + volatility filter + volume confirmation
# - Exit: Opposite Donchian breakout or ATR drops below 0.5 * 20-period MA (volatility collapse)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian captures trends, volatility filter avoids false breakouts in ranging markets, volume confirms conviction
# - Target: 100-180 trades over 4 years (25-45/year) within fee drag limits for 4h timeframe

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for indicators
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian Channel (20-period)
    lookback = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        if not np.isnan(high_4h[i-lookback+1:i+1]).any() and not np.isnan(low_4h[i-lookback+1:i+1]).any():
            upper_channel[i] = np.max(high_4h[i-lookback+1:i+1])
            lower_channel[i] = np.min(low_4h[i-lookback+1:i+1])
    
    # Calculate 4h volume moving average (20-period)
    volume_ma_20_4h = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        if not np.isnan(volume_4h[i-lookback+1:i+1]).any():
            volume_ma_20_4h[i] = np.mean(volume_4h[i-lookback+1:i+1])
    
    # Calculate 1d ATR(14) for volatility filter
    atr_lookback = 14
    tr1 = np.maximum(np.maximum(high_1d - low_1d,
                               np.abs(np.roll(high_1d, 1) - low_1d)),
                    np.abs(np.roll(low_1d, 1) - high_1d))
    
    atr_1d = np.full(len(tr1), np.nan)
    for i in range(atr_lookback, len(tr1)):
        if not np.isnan(tr1[i-atr_lookback:i]).any():
            atr_1d[i] = np.mean(tr1[i-atr_lookback:i])
    
    # Calculate 1d ATR moving average (20-period) for volatility regime
    atr_ma_20_1d = np.full(len(atr_1d), np.nan)
    for i in range(19, len(atr_1d)):  # 20-period MA
        if not np.isnan(atr_1d[i-19:i+1]).any():
            atr_ma_20_1d[i] = np.mean(atr_1d[i-19:i+1])
    
    # Align all HTF indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(volume_ma_20_4h[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.8 * 20-period MA of ATR (avoid low volatility breakouts)
        volatility_filter = atr_1d_aligned[i] > 0.8 * atr_ma_20_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.2 * 20-period MA of volume
        volume_confirm = volume_4h[i] > 1.2 * volume_ma_20_4h[i]
        
        # Donchian breakout conditions
        breakout_up = close_4h[i] > upper_channel[i-1]  # Break above previous upper channel
        breakout_down = close_4h[i] < lower_channel[i-1]  # Break below previous lower channel
        
        # Exit conditions: opposite breakout OR volatility collapse (ATR < 0.5 * MA)
        exit_long = breakout_down or (atr_1d_aligned[i] < 0.5 * atr_ma_20_1d_aligned[i])
        exit_short = breakout_up or (atr_1d_aligned[i] < 0.5 * atr_ma_20_1d_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish breakout + volatility filter + volume confirmation
            if breakout_up and volatility_filter and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish breakout + volatility filter + volume confirmation
            elif breakout_down and volatility_filter and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: opposite breakout or volatility collapse
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals