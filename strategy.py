#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ATR volatility filter
# - Long: Price breaks above Camarilla H3 level + 1d volume > 1.5x 20-period MA + 1d ATR(14) > 1.2x ATR MA(50)
# - Short: Price breaks below Camarilla L3 level + same volume/volatility conditions
# - Exit: Price returns to Camarilla pivot point (PP)
# - Uses 1d HTF for volume and volatility to ensure breakouts occur with institutional participation
# - Volatility filter avoids low-momentum environments; volume confirmation reduces false breakouts
# - Works in bull/bear: breakouts with volume/volatility confirmation capture strong moves while avoiding chop
# - Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag

name = "4h_1d_camarilla_breakout_vol_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # PP = (high + low + close)/3
    rng = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_h3 = close_1d + 1.1 * rng * 1.1 / 4.0
    camarilla_l3 = close_1d - 1.1 * rng * 1.1 / 4.0
    camarilla_h4 = close_1d + 1.1 * rng * 1.1 / 2.0
    camarilla_l4 = close_1d - 1.1 * rng * 1.1 / 2.0
    
    # Calculate 1d ATR(14)
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 50-period MA of 1d ATR for volatility regime filter
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period (need at least 60 for ATR14 and ATR MA50)
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get aligned 1d data for current 4h bar (completed 1d bar)
        atr_14_current = atr_14_aligned[i]
        atr_ma_50_current = atr_ma_50_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        camarilla_pp_current = camarilla_pp[i]
        camarilla_h3_current = camarilla_h3[i]
        camarilla_l3_current = camarilla_l3[i]
        
        # Volatility condition: current 1d ATR > 1.2x 50-period MA of ATR
        vol_condition = atr_14_current > 1.2 * atr_ma_50_current
        
        # Volume spike condition: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d_current > 1.5 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Camarilla H3 + volatility condition + volume spike
            if (close_price > camarilla_h3_current and vol_condition and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Camarilla L3 + volatility condition + volume spike
            elif (close_price < camarilla_l3_current and vol_condition and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to Camarilla pivot point (PP)
            if position == 1 and close_price <= camarilla_pp_current:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close_price >= camarilla_pp_current:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals