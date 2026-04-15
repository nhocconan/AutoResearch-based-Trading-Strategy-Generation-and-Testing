#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Long when price breaks above Camarilla R4 + 1d close > EMA34 + volume > 1.5x 20-period avg
# Short when price breaks below Camarilla S4 + 1d close < EMA34 + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 1d EMA34 provides trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~20-40 trades/year to minimize fee drag on 6h timeframe.
# Camarilla levels calculated from prior 1d OHLC.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 34 + 5  # EMA34 + buffer
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from prior 1d candle
        # Need at least one completed 1d bar to calculate levels
        if i < 24:  # Need at least 24 6h bars (1d) to get prior day
            signals[i] = 0.0
            continue
            
        # Get index of prior completed 1d bar in 6h data
        # Each 1d = 4 6h bars, so prior day ends at index i - (i % 4) - 4
        prior_day_end_idx = i - (i % 4) - 4
        if prior_day_end_idx < 0:
            signals[i] = 0.0
            continue
            
        # Get OHLC of prior completed 1d bar
        phigh = high[prior_day_end_idx - 3:prior_day_end_idx + 1].max()
        plow = low[prior_day_end_idx - 3:prior_day_end_idx + 1].min()
        pclose = close[prior_day_end_idx]
        
        # Calculate Camarilla levels
        range_val = phigh - plow
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        camarilla_r4 = pclose + (range_val * 1.1 / 2)
        camarilla_s4 = pclose - (range_val * 1.1 / 2)
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        if i >= 20:
            vol_sma_20 = np.mean(volume[i-20:i])
            vol_confirm = volume[i] > (vol_sma_20 * 1.5)
        else:
            vol_confirm = False
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R4 (close > R4)
        # 2. 1d EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        if (close[i] > camarilla_r4) and \
           (close[i] > ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S4 (close < S4)
        # 2. 1d EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s4) and \
             (close[i] < ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_CamarillaR4S4_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0