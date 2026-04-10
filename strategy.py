#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d ATR regime filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 1d ATR(14) > 1d ATR(50) (expanding daily volatility)
# - Short when price breaks below Camarilla L3 level AND 1d ATR(14) > 1d ATR(50)
# - Volume confirmation: 4h volume > 1.5x 20-period 4h volume SMA
# - Exit: price returns to Camarilla Pivot Point (PP) or opposite breakout with volume
# - Uses 1d ATR for regime filter to avoid whipsaw in ranging markets
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Position sizing: 0.25 discrete level

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
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track entry price for stoploss (optional, using signal=0 for exit)
    entry_price = np.full(n, np.nan)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d ATR to 4h timeframe (completed 1d bar only)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla: PP = (H+L+C)/3, Range = H-L
    # H4 = PP + Range * 1.1/2, L4 = PP - Range * 1.1/2
    # H3 = PP + Range * 1.1/4, L3 = PP - Range * 1.1/4
    # H2 = PP + Range * 1.1/6, L2 = PP - Range * 1.1/6
    # H1 = PP + Range * 1.1/12, L1 = PP - Range * 1.1/12
    
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    camarilla_pp = typical_price_1d
    camarilla_h3 = camarilla_pp + (range_1d * 1.1 / 4)
    camarilla_l3 = camarilla_pp - (range_1d * 1.1 / 4)
    camarilla_h4 = camarilla_pp + (range_1d * 1.1 / 2)
    camarilla_l4 = camarilla_pp - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_50_1d_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_pp_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 1d ATR(14) > 1d ATR(50) (expanding daily volatility regime)
        vol_regime = atr_14_1d_aligned[i] > atr_50_1d_aligned[i]
        
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
            # Exit on return to Camarilla PP or opposite breakout with volume
            exit_condition = (close[i] < camarilla_pp_aligned[i]) or \
                           (breakout_down and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit on return to Camarilla PP or opposite breakout with volume
            exit_condition = (close[i] > camarilla_pp_aligned[i]) or \
                           (breakout_up and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals