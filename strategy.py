#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume spike and ATR volatility filter
# - Primary: 12h price breaks above/below Camarilla H3/L3 levels (based on prior 1d candle)
# - HTF Confirmation: 1d volume > 2.0x 20-period MA (strong conviction) + ATR(14) < 0.03 * price (low volatility regime)
# - Long: Price > Camarilla H3 + volume spike + low volatility
# - Short: Price < Camarilla L3 + volume spike + low volatility
# - Exit: Price reverts to Camarilla Pivot point (mean reversion) or volatility spikes (ATR > 0.05 * price)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Camarilla pivots capture institutional levels, volume confirms breakout strength, volatility filter avoids choppy markets
# - Target: 80-120 total trades over 4 years (20-30/year) to stay within fee drag limits for 12h timeframe

name = "12h_1d_camarilla_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for indicators
        return np.zeros(n)
    
    # Pre-compute 12h data
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate 1d ATR (14-period) for volatility filter
    tr = np.maximum(np.maximum(high_1d - low_1d,
                              np.abs(np.roll(high_1d, 1) - low_1d)),
                   np.abs(np.roll(low_1d, 1) - high_1d))
    atr_14_1d = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_14_1d[i] = np.mean(tr[i-13:i+1])
    
    # Align all HTF indicators to 12h timeframe
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from second bar to have previous day for pivot calc
        # Skip if any required data is invalid
        if (np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned to 12h)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 2.0x 20-period MA (strong conviction)
        volume_confirm = volume_1d_aligned[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        # Volatility filter: ATR(14) < 0.03 * price (low volatility regime)
        vol_filter = atr_14_1d_aligned[i] < 0.03 * close_1d_aligned[i]
        
        # Calculate Camarilla pivot levels from previous 1d candle
        # Only calculate at the start of each 1d candle (00:00 UTC)
        if i > 0 and prices['open_time'].iloc[i].date() != prices['open_time'].iloc[i-1].date():
            # Previous 1d candle
            prev_high = high_1d[i-1] if not np.isnan(high_1d[i-1]) else np.nan
            prev_low = low_1d[i-1] if not np.isnan(low_1d[i-1]) else np.nan
            prev_close = close_1d[i-1] if not np.isnan(close_1d[i-1]) else np.nan
            
            if not (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
                # Camarilla levels
                camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 6
                camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 6
                camarilla_pivot = (prev_high + prev_low + prev_close) / 3
                camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 4
                camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 4
            else:
                camarilla_h3 = camarilla_l3 = camarilla_pivot = camarilla_h4 = camarilla_l4 = np.nan
        else:
            # Carry forward previous day's levels
            if i == 1:
                camarilla_h3 = camarilla_l3 = camarilla_pivot = camarilla_h4 = camarilla_l4 = np.nan
        
        # Skip if pivot levels not available
        if np.isnan(camarilla_h3) or np.isnan(camarilla_l3) or np.isnan(camarilla_pivot):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout signals
        camarilla_up = close_12h[i] > camarilla_h3
        camarilla_down = close_12h[i] < camarilla_l3
        
        # Exit conditions: Price reverts to Camarilla Pivot point OR volatility spikes (ATR > 0.05 * price)
        exit_long = (close_12h[i] < camarilla_pivot) or (atr_14_1d_aligned[i] > 0.05 * close_1d_aligned[i])
        exit_short = (close_12h[i] > camarilla_pivot) or (atr_14_1d_aligned[i] > 0.05 * close_1d_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Camarilla breakout up + volume confirmation + low volatility
            if camarilla_up and volume_confirm and vol_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: Camarilla breakout down + volume confirmation + low volatility
            elif camarilla_down and volume_confirm and vol_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price reverts to Camarilla Pivot OR volatility spikes
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