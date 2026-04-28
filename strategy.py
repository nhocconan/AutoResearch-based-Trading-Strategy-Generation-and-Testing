#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR filter
# Long when price breaks above Donchian upper band AND volume > 1.5x 20-bar avg AND ATR(14) > 0.01*close
# Short when price breaks below Donchian lower band AND volume > 1.5x 20-bar avg AND ATR(14) > 0.01*close
# Exits when price touches the opposite Donchian band or ATR(14) < 0.005*close (volatility collapse)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year on 4h.
# Works in bull markets by trading breakouts with volume confirmation, works in bear by requiring 
# sufficient volatility (ATR filter) to avoid whipsaws in low-volume ranging markets.

name = "4h_Donchian20_Volume_ATR_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # ATR(14) calculation for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filters: ATR > 0.01*close for entry, ATR < 0.005*close for exit (vol collapse)
    vol_filter_entry = atr > 0.01 * close
    vol_filter_exit = atr < 0.005 * close
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        vol_entry = vol_filter_entry[i]
        vol_exit = vol_filter_exit[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above upper band AND volume confirmation AND sufficient volatility
            if curr_close > upper and vol_conf and vol_entry:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower band AND volume confirmation AND sufficient volatility
            elif curr_close < lower and vol_conf and vol_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price touches lower band OR volatility collapses
            if curr_close < lower or vol_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price touches upper band OR volatility collapses
            if curr_close > upper or vol_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals