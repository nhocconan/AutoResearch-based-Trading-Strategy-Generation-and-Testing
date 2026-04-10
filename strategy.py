#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR-based stoploss
# - Long when price breaks above Camarilla R4 (1d) AND 1d volume > 1.8x 24-bar avg
# - Short when price breaks below Camarilla S4 (1d) AND 1d volume > 1.8x 24-bar avg
# - Exit when price returns to Camarilla PP (pivot point) from 1d OR ATR stoploss hit (2.5x ATR)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla levels provide precise support/resistance; volume confirms institutional participation
# - ATR stoploss manages risk in both bull and bear markets

name = "12h_1d_camarilla_breakout_volume_stop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from 1d data (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2.0)
    camarilla_s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 1d volume confirmation: > 1.8x 24-period average
    volume_1d = df_1d['volume'].values
    volume_24_avg = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    vol_spike_1d = volume_1d > (1.8 * volume_24_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute ATR for stoploss (using 1h data for better responsiveness)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        atr_1h = np.full(n, np.nan)
    else:
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        close_1h = df_1h['close'].values
        tr1 = high_1h - low_1h
        tr2 = np.abs(high_1h - np.roll(close_1h, 1))
        tr3 = np.abs(low_1h - np.roll(close_1h, 1))
        tr1[0] = np.nan
        tr2[0] = np.nan
        tr3[0] = np.nan
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        atr_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_1h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(atr_1h_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Camarilla R4 AND 1d volume spike
            if (prices['high'].iloc[i] > camarilla_r4_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                atr_val = atr_1h_aligned[i]
                stop_price = entry_price - 2.5 * atr_val if not np.isnan(atr_val) else entry_price * 0.97
                signals[i] = 0.25
            # Short when price breaks below Camarilla S4 AND 1d volume spike
            elif (prices['low'].iloc[i] < camarilla_s4_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                atr_val = atr_1h_aligned[i]
                stop_price = entry_price + 2.5 * atr_val if not np.isnan(atr_val) else entry_price * 1.03
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - check for exits
            exit_signal = False
            current_price = prices['close'].iloc[i]
            
            # Check ATR stoploss
            if position == 1:  # Long position
                if current_price <= stop_price:
                    exit_signal = True
                # Also exit when price returns to Camarilla pivot point (mean reversion)
                elif current_price <= camarilla_pp_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if current_price >= stop_price:
                    exit_signal = True
                # Also exit when price returns to Camarilla pivot point (mean reversion)
                elif current_price >= camarilla_pp_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals