#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume regime filter and session timing.
# Long when price breaks above 4h Donchian(20) high AND 1d volume > 1.5x 20-day average AND session 08-20 UTC.
# Short when price breaks below 4h Donchian(20) low AND 1d volume > 1.5x 20-day average AND session 08-20 UTC.
# Exit on opposite Donchian break or ATR stop (2*ATR). Uses discrete size 0.20.
# Target: 80-150 total trades over 4 years (20-38/year). Works in bull/bear via volume regime filter
# that avoids low-volume false breakouts. Uses 4h for structure, 1h only for entry timing precision.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    highest_high_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # === 1d Indicators: Volume Regime Filter ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_regime = vol_1d > (1.5 * vol_ma_1d_aligned)  # High volume regime
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(df_4h['high']).diff()
    tr2 = pd.Series(df_4h['low']).diff().abs()
    tr3 = pd.Series(df_4h['close']).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(volume_regime[i]) or np.isnan(atr_4h_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_reg = volume_regime[i]
        atr_val = atr_4h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below 4h Donchian low
            if price < donchian_low_4h_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above 4h Donchian high
            if price > donchian_high_4h_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 4h Donchian high AND volume regime AND session
            if price > donchian_high_4h_aligned[i] and vol_reg:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below 4h Donchian low AND volume regime AND session
            elif price < donchian_low_4h_aligned[i] and vol_reg:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_Donchian20_1dVolumeRegime_Session_V1"
timeframe = "1h"
leverage = 1.0