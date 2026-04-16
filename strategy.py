#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Stochastic Oscillator with 1d RSI filter and volume confirmation.
# Long when Stochastic %K crosses above 20 AND RSI(14) > 50 AND volume > 1.5x 20-period average.
# Short when Stochastic %K crosses below 80 AND RSI(14) < 50 AND volume > 1.5x 20-period average.
# Exit on opposite Stochastic cross or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture mean-reversion bounces in ranging markets
# while avoiding counter-trend trades via RSI filter. Target: 75-200 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Stochastic Oscillator (14,3,3) ===
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_k_prev = np.roll(stoch_k, 1)
    stoch_k_prev[0] = np.nan
    
    # === 1d Indicators: RSI(14) and Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    rsi_period = 14
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
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
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for volume MA)
    warmup = 30
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(stoch_k[i]) or np.isnan(stoch_k_prev[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_4h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Stochastic %K crosses below 80 (overbought)
            if stoch_k[i] < 80 and stoch_k_prev[i] >= 80:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Stochastic %K crosses above 20 (oversold)
            if stoch_k[i] > 20 and stoch_k_prev[i] <= 20:
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
            # LONG: Stochastic %K crosses above 20 AND RSI > 50 AND volume spike
            if (stoch_k[i] > 20 and stoch_k_prev[i] <= 20 and 
                rsi_1d_aligned[i] > 50 and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Stochastic %K crosses below 80 AND RSI < 50 AND volume spike
            elif (stoch_k[i] < 80 and stoch_k_prev[i] >= 80 and 
                  rsi_1d_aligned[i] < 50 and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Stochastic_1dRSI_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0