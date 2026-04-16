#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND 1w EMA200 up trending AND volume > 1.5x 20-period average.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND 1w EMA200 down trending AND volume > 1.5x 20-period average.
# Exit on opposite Alligator alignment or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to catch trends with Alligator alignment while filtering chop via 1w EMA200 slope.
# Target: 30-100 total trades over 4 years (7-25/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Williams Alligator (SMAs of median price) ===
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    # Alligator lines: Jaw (13-period SMA, 8 bars offset), Teeth (8-period SMA, 5 bars offset), Lips (5-period SMA, 3 bars offset)
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # === 1w Indicators: EMA200 slope for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema200_1w_raw = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean()
    ema200_1w = ema200_1w_raw.values
    # Slope: positive if current > previous 5 periods ago
    ema200_slope_raw = ema200_1w - np.roll(ema200_1w, 5)
    ema200_slope_raw[:5] = 0  # avoid NaN
    
    # === 1d Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_raw = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ma = vol_ma_raw.values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 1d ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_raw = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to LTF
    jaw = align_htf_to_ltf(prices, pd.DataFrame({'close': median_price}), jaw_raw)  # using median_price as close proxy for alignment
    teeth = align_htf_to_ltf(prices, pd.DataFrame({'close': median_price}), teeth_raw)
    lips = align_htf_to_ltf(prices, pd.DataFrame({'close': median_price}), lips_raw)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    ema200_slope_aligned = align_htf_to_ltf(prices, df_1w, ema200_slope_raw)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 200 periods needed for EMA200)
    warmup = 250
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(ema200_slope_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Alligator alignment turns bearish (jaws > teeth > lips)
            if jaw[i] > teeth[i] and teeth[i] > lips[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Alligator alignment turns bullish (jaws < teeth < lips)
            if jaw[i] < teeth[i] and teeth[i] < lips[i]:
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
            # LONG: Bullish Alligator alignment AND price > lips AND 1w EMA200 up trending AND volume spike
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and 
                price > lips[i] and 
                ema200_slope_aligned[i] > 0 and 
                vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bearish Alligator alignment AND price < lips AND 1w EMA200 down trending AND volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  price < lips[i] and 
                  ema200_slope_aligned[i] < 0 and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA200Slope_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0