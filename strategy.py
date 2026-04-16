#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum breakout with 4h trend filter (EMA50) and 1d volume confirmation.
# Long when 1h price crosses above 4h EMA50 AND 1h RSI > 50 AND 1d volume > 1.5x 20-period average.
# Short when 1h price crosses below 4h EMA50 AND 1h RSI < 50 AND 1d volume > 1.5x 20-period average.
# Exit on opposite cross or ATR-based stoploss (2*ATR from entry).
# Uses 4h EMA50 for trend direction, 1h for entry timing, 1d volume for conviction.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: RSI(14) ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    rsi_prev = np.roll(rsi, 1)
    rsi_prev[0] = np.nan
    
    # === 4h Indicators: EMA50 ===
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d Indicators: Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1h_raw = pd.Series(tr_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(rsi_prev[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_1h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        rsi_val = rsi[i]
        rsi_prev_val = rsi_prev[i]
        ema_val = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below 4h EMA50
            if price < ema_val and ema_50_4h_aligned[i-1] <= close[i-1]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above 4h EMA50
            if price > ema_val and ema_50_4h_aligned[i-1] >= close[i-1]:
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
            # LONG: Price crosses above 4h EMA50 AND RSI > 50 AND volume spike
            if (price > ema_val and ema_50_4h_aligned[i-1] <= close[i-1] and 
                rsi_val > 50 and vol_spike):
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Price crosses below 4h EMA50 AND RSI < 50 AND volume spike
            elif (price < ema_val and ema_50_4h_aligned[i-1] >= close[i-1] and 
                  rsi_val < 50 and vol_spike):
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_EMA50_RSI_VolumeSpike_SessionFilter_V1"
timeframe = "1h"
leverage = 1.0