#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with daily volume confirmation and ATR regime filter
# - Long when price breaks above Camarilla H3 level AND daily volume > 1.5x 20-day volume SMA AND ATR(14) > ATR(50)
# - Short when price breaks below Camarilla L3 level AND daily volume > 1.5x 20-day volume SMA AND ATR(14) > ATR(50)
# - Exit: price returns to Camarilla H4/L4 levels or opposite breakout with volume confirmation
# - Position sizing: 0.25 discrete level
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Combines Camarilla pivot structure (proven ETH edge) with volatility filter to reduce whipsaw in ranging markets

name = "4h_1d_camarilla_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate ATR(14) and ATR(50) for volatility regime filter (4h)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 20-period volume SMA for confirmation (4h)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load daily HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from daily data
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4, etc.
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    daily_range = daily_high - daily_low
    camarilla_h4 = daily_close + 1.1 * daily_range * 1.1 / 2
    camarilla_h3 = daily_close + 1.1 * daily_range * 1.1 / 4
    camarilla_l3 = daily_close - 1.1 * daily_range * 1.1 / 4
    camarilla_l4 = daily_close - 1.1 * daily_range * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (wait for daily bar to close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Track entry extreme for exit logic
    entry_price = np.full(n, np.nan)
    
    for i in range(20, n):  # Start after 20-bar warmup for volume SMA
        # Skip if any required data is invalid
        if (np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR(14) > ATR(50) (expanding volatility regime)
        vol_regime = atr_14[i] > atr_50[i]
        
        # Volume confirmation: 4h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # Break above previous H3
        breakout_down = close[i] < camarilla_l3_aligned[i-1]  # Break below previous L3
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_regime and vol_confirm:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]
            elif breakout_down and vol_regime and vol_confirm:
                position = -1
                signals[i] = -0.25
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit on return to H4 level or opposite breakout with volume confirmation
            exit_condition = (close[i] < camarilla_h4_aligned[i]) or \
                           (breakout_down and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit on return to L4 level or opposite breakout with volume confirmation
            exit_condition = (close[i] > camarilla_l4_aligned[i]) or \
                           (breakout_up and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals