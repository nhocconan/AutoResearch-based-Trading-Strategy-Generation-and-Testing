#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Keltner Channel breakout with 1d ADX trend strength filter and volume confirmation.
# Keltner Channel uses ATR (true volatility) instead of fixed standard deviation, adapting to market conditions.
# We trade breakouts only when ADX > 25 (strong trend) to avoid false signals in ranging markets.
# Volume confirmation ensures breakouts are supported by participation.
# Works in bull/bear markets: ADX filter avoids ranging markets, Keltner adapts to volatility regimes.
# Target: 20-40 trades/year per symbol.
name = "12h_Keltner_ADX25_Volume_Breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros(len(high))
        atr[period] = np.nansum(tr[1:period+1]) if period < len(tr) else 0
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (np.array([np.nansum(plus_dm[max(1, i-period+1):i+1]) if i >= period else 0 for i in range(len(high))]) / np.where(atr == 0, 1, atr))
        minus_di = 100 * (np.array([np.nansum(minus_dm[max(1, i-period+1):i+1]) if i >= period else 0 for i in range(len(high))]) / np.where(atr == 0, 1, atr))
        
        dx = np.zeros(len(high))
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
        
        adx = np.zeros(len(high))
        if len(dx) >= 2*period:
            adx[2*period-1] = np.nanmean(dx[period:2*period]) if np.any(~np.isnan(dx[period:2*period])) else 0
            for i in range(2*period, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Keltner Channel (20, 2.0) on 12h
    kc_period = 20
    kc_mult = 2.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # ATR
    atr_kc = np.zeros(len(tr))
    atr_kc[kc_period] = np.mean(tr[1:kc_period+1]) if kc_period < len(tr) else 0
    for i in range(kc_period+1, len(tr)):
        atr_kc[i] = (atr_kc[i-1] * (kc_period-1) + tr[i]) / kc_period
    
    # Middle line (EMA)
    ema_middle = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    # Upper and lower bands
    upper_keltner = ema_middle + (kc_mult * atr_kc)
    lower_keltner = ema_middle - (kc_mult * atr_kc)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ADX to 12h
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kc_period, 20)  # Ensure Keltner and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_middle[i]) or np.isnan(atr_kc[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(vol_ma_20[i]) or np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_keltner[i]
        lower = lower_keltner[i]
        adx_val = adx_14_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        # Breakout conditions
        bullish_breakout = price > upper  # Price breaks above upper band
        bearish_breakout = price < lower  # Price breaks below lower band
        
        if position == 0:
            # Look for entry when ADX indicates strong trend (>25) and volume confirms
            if adx_val > 25 and bullish_breakout and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif adx_val > 25 and bearish_breakout and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to middle band (mean reversion)
            middle = ema_middle[i]
            if price < middle:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to middle band
            middle = ema_middle[i]
            if price > middle:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals