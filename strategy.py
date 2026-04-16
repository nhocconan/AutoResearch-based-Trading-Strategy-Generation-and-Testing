#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR-based volatility filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d ATR(14) > 1.5x 50-period average (high volatility regime) AND 1d volume > 1.2x 20-period average.
# Short when price breaks below Donchian(20) low AND same volatility/volume conditions.
# Exit on ATR-based stoploss (2*ATR from entry) or opposite Donchian break.
# Uses discrete position size 0.25. Targets high-probability breakouts during volatile sessions to avoid false breakouts in low volatility.
# Timeframe: 4h. HTF: 1d for volatility and volume filters.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: ATR and Volume ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # True Range for 1d ATR
    tr1_1d = pd.Series(high_1d).diff()
    tr2_1d = pd.Series(low_1d).diff().abs()
    tr3_1d = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d_raw = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 1d ATR 50-period average (regime filter)
    atr_ma_1d = pd.Series(atr_1d_raw).rolling(window=50, min_periods=50).mean().values
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # 1d Volume 20-period average
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volatility regime: ATR > 1.5x 50-period average
    vol_regime = atr_1d_raw > (1.5 * atr_ma_1d_aligned)
    
    # Volume confirmation: volume > 1.2x 20-period average
    volume_spike = volume > (1.2 * vol_ma_1d_aligned)
    
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
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ATR MA)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_regime[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_4h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_reg = vol_regime[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low (opposite breakout)
            if price < low_roll[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high (opposite breakout)
            if price > high_roll[i]:
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
            # LONG: Price breaks above Donchian high AND high volatility regime AND volume spike
            if price > high_roll[i] and vol_reg and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian low AND high volatility regime AND volume spike
            elif price < low_roll[i] and vol_reg and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dATR_Volume_V1"
timeframe = "4h"
leverage = 1.0