#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R1/S1 breakout with 1w trend filter and volume confirmation.
# Long when price breaks above R1 AND 1w close > 1w EMA50 AND 1d volume > 2x 20-period average.
# Short when price breaks below S1 AND 1w close < 1w EMA50 AND 1d volume > 2x 20-period average.
# Exit on ATR-based stoploss (2*ATR from entry) or opposite Camarilla breakout.
# Uses discrete position size 0.25. Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    # Use previous day's OHLC to calculate today's levels
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    # === 1w Indicators: EMA50 Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d Volume Spike ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
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
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema_50_1w_aligned[i]) or
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
            # Exit if price breaks below S1 (opposite breakout)
            if price < s1[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above R1 (opposite breakout)
            if price > r1[i]:
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
            # LONG: Price breaks above R1 AND 1w close > 1w EMA50 AND volume spike
            if (price > r1[i] and 
                df_1w['close'].values[-1] > ema_50_1w[-1] if len(df_1w) > 0 else False and  # Use latest 1w close
                vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below S1 AND 1w close < 1w EMA50 AND volume spike
            elif (price < s1[i] and 
                  df_1w['close'].values[-1] < ema_50_1w[-1] if len(df_1w) > 0 else False and  # Use latest 1w close
                  vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wEMA50_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0