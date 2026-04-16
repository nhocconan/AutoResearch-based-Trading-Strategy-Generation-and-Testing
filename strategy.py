#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and choppiness regime filter.
# Long when Williams %R < -80 (oversold) AND 1d volume > 1.3x 20-period average AND chop > 61.8 (ranging market).
# Short when Williams %R > -20 (overbought) AND 1d volume > 1.3x 20-period average AND chop > 61.8.
# Exit when Williams %R crosses -50 (mean reversion completion) or ATR-based stoploss (1.5*ATR).
# Uses discrete position size 0.25. Designed to capture mean reversion in ranging markets with volume confirmation.
# Works in both bull and bear markets by requiring volume confirmation and choppiness regime filter.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Williams %R (14-period) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # === 1d Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.3 * vol_ma_1d_aligned)
    
    # === 1d Indicators: Choppiness Index (14-period) ===
    atr_1d = []
    tr_list = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        else:
            tr = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
        tr_list.append(tr)
    
    tr_1d = pd.Series(tr_list)
    atr_1d_sum = tr_1d.rolling(window=14, min_periods=14).sum().values
    max_high_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    chop_denom = max_high_1d - min_low_1d
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_1d = 100 * np.log10(atr_1d_sum / np.log10(chop_denom) * np.sqrt(14)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(atr_4h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_val = chop_1d_aligned[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50 (mean reversion complete)
            if williams_r[i] > -50:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50 (mean reversion complete)
            if williams_r[i] < -50:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND volume spike AND chop > 61.8 (ranging)
            if williams_r[i] < -80 and vol_spike and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R > -20 (overbought) AND volume spike AND chop > 61.8 (ranging)
            elif williams_r[i] > -20 and vol_spike and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_WilliamsR_Oversold_1dVolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0