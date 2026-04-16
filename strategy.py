#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d CCI(20) with 1w EMA trend filter and volume confirmation.
# Long when CCI(20) crosses above -100 AND price > 1w EMA50 AND 1d volume > 1.5x 20-period average.
# Short when CCI(20) crosses below +100 AND price < 1w EMA50 AND 1d volume > 1.5x 20-period average.
# Exit on opposite CCI cross or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Weekly trend filter avoids counter-trend trades in strong trends.
# Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: CCI(20) ===
    tp = (high + low + close) / 3.0
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    md_tp = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - ma_tp) / (0.015 * md_tp)
    cci_prev = np.roll(cci, 1)
    cci_prev[0] = np.nan
    
    # === 1w Indicators: EMA50 ===
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d Volume Spike ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # === 1d ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d_raw = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or np.isnan(cci_prev[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_1d_raw[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1d_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if CCI crosses below +100 (momentum loss)
            if cci[i] < 100 and cci_prev[i] >= 100:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if CCI crosses above -100 (momentum loss)
            if cci[i] > -100 and cci_prev[i] <= -100:
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
            # LONG: CCI crosses above -100 AND price > 1w EMA50 AND volume spike
            if (cci[i] > -100 and cci_prev[i] <= -100 and 
                price > ema_50_1w_aligned[i] and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: CCI crosses below +100 AND price < 1w EMA50 AND volume spike
            elif (cci[i] < 100 and cci_prev[i] >= 100 and 
                  price < ema_50_1w_aligned[i] and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_CCI20_1wEMA50_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0