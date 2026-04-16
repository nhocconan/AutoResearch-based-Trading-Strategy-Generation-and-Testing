#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX with 12h EMA trend filter and volume confirmation.
# Long when TRIX crosses above zero AND price > 12h EMA34 AND 12h volume > 1.3x 20-period average.
# Short when TRIX crosses below zero AND price < 12h EMA34 AND 12h volume > 1.3x 20-period average.
# Exit on ATR-based stoploss (2*ATR from entry) or opposite TRIX cross.
# Uses discrete position size 0.25. TRIX is a momentum oscillator that filters noise well in ranging markets.
# Volume confirmation ensures institutional participation. 12h EMA34 provides multi-timeframe trend alignment.
# Target: 75-200 total trades over 4 years (19-50/year) with strong performance in both bull and bear regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: TRIX (15-period) ===
    # TRIX = triple-smoothed EMA of percentage price change
    roc = pd.Series(close).pct_change().values
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3 * 100  # scale for readability
    trix_prev = np.roll(trix, 1)
    trix_prev[0] = np.nan
    
    # === 12h Indicators: EMA34 and Volume Spike ===
    df_12h = get_htf_data(prices, '12h')
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.3 * vol_ma_12h_aligned)
    
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
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA34)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(trix[i]) or np.isnan(trix_prev[i]) or np.isnan(ema_34_12h_aligned[i]) or
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
            # Exit if TRIX crosses below zero (momentum loss)
            if trix[i] < 0 and trix_prev[i] >= 0:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if TRIX crosses above zero (momentum loss)
            if trix[i] > 0 and trix_prev[i] <= 0:
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
            # LONG: TRIX crosses above zero AND price > 12h EMA34 AND volume spike
            if (trix[i] > 0 and trix_prev[i] <= 0 and 
                price > ema_34_12h_aligned[i] and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: TRIX crosses below zero AND price < 12h EMA34 AND volume spike
            elif (trix[i] < 0 and trix_prev[i] >= 0 and 
                  price < ema_34_12h_aligned[i] and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_TRIX_12hEMA34_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0