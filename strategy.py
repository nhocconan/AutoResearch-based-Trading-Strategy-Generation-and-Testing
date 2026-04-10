#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and ATR stoploss
# - Long when price breaks above Camarilla H3 level AND 12h volume > 1.3x 20-period volume SMA
# - Short when price breaks below Camarilla L3 level AND 12h volume > 1.3x 20-period volume SMA
# - Exit: ATR-based trailing stop (2.5x ATR from extreme) or price retracement to Camarilla H4/L4
# - Uses actual Camarilla formula: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
# - Position sizing: 0.25 discrete level to minimize fee impact while maintaining profitability
# - Target: 30-60 trades/year on 4h timeframe to stay within fee drag limits
# - Camarilla levels provide institutional support/resistance, volume confirms breakout strength

name = "4h_12h_camarilla_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla levels from daily data
    # True Camarilla formula: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    daily_range = df_1d['high'].values - df_1d['low'].values
    camarilla_h4 = df_1d['close'].values + 1.1 * daily_range * 1.1 / 2.0
    camarilla_h3 = df_1d['close'].values + 1.1 * daily_range * 1.1 / 4.0
    camarilla_l3 = df_1d['close'].values - 1.1 * daily_range * 1.1 / 4.0
    camarilla_l4 = df_1d['close'].values - 1.1 * daily_range * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate ATR(10) for stoploss
    atr_period = 10
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate 12h volume SMA for confirmation
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Track extreme prices for trailing stop
    long_extreme = np.full(n, np.nan)
    short_extreme = np.full(n, np.nan)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(volume_sma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.3x 20-period volume SMA
        # Each 12h bar = 48 4h bars
        idx_12h = i // 48
        if idx_12h < len(volume_12h):
            vol_confirm = volume_12h[idx_12h] > 1.3 * volume_sma_20_12h_aligned[i]
        else:
            vol_confirm = False
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3_aligned[i]  # Break above H3
        breakout_down = close[i] < camarilla_l3_aligned[i]  # Break below L3
        
        # Update extremes for trailing stop
        if position == 1:  # Long position
            if np.isnan(long_extreme[i-1]):
                long_extreme[i] = close[i]
            else:
                long_extreme[i] = max(long_extreme[i-1], close[i])
        elif position == -1:  # Short position
            if np.isnan(short_extreme[i-1]):
                short_extreme[i] = close[i]
            else:
                short_extreme[i] = min(short_extreme[i-1], close[i])
        else:
            long_extreme[i] = np.nan
            short_extreme[i] = np.nan
        
        # ATR-based trailing stop conditions
        stop_long = False
        stop_short = False
        
        if position == 1 and not np.isnan(long_extreme[i]):
            stop_long = close[i] < long_extreme[i] - 2.5 * atr[i]
        elif position == -1 and not np.isnan(short_extreme[i]):
            stop_short = close[i] > short_extreme[i] + 2.5 * atr[i]
        
        # Camarilla retracement exit (to H4/L4 levels)
        exit_long = close[i] < camarilla_h4_aligned[i]
        exit_short = close[i] > camarilla_l4_aligned[i]
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if stop_long or exit_long:
                position = 0
                signals[i] = 0.0
                long_extreme[i] = np.nan  # Reset extreme
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if stop_short or exit_short:
                position = 0
                signals[i] = 0.0
                short_extreme[i] = np.nan  # Reset extreme
            else:
                signals[i] = -0.25
    
    return signals