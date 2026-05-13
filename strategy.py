# Solution: 1D RSI Divergence with Volume Confirmation and Volatility Filter
# Hypothesis: RSI divergence (bullish/bearish) at extremes combined with volume confirmation
# and volatility filter (ATR-based) works in both bull and bear markets by catching
# exhaustion moves. Uses daily timeframe for higher reliability and lower trade frequency.
# Target: 10-25 trades/year to minimize fee drag.

#!/usr/bin/env python3
name = "1D_RSI_Divergence_Volume_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE (same as primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate RSI(14) on 1D
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        # First values
        if len(gain) >= period:
            avg_gain[period-1] = np.nanmean(gain[1:period])
            avg_loss[period-1] = np.nanmean(loss[1:period])
            
            # Wilder's smoothing
            for i in range(period, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.full_like(close, np.nan)
        valid = ~np.isnan(avg_loss) & (avg_loss != 0)
        rs[valid] = avg_gain[valid] / avg_loss[valid]
        
        rsi = np.full_like(close, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate ATR(14) for volatility filter
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    rsi_1d = calculate_rsi(close_1d, 14)
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align indicators to 1D timeframe (no alignment needed as same timeframe)
    rsi_1d_aligned = rsi_1d
    atr_1d_aligned = atr_1d
    
    # Calculate 20-period SMA for trend filter
    close_s = pd.Series(close_1d)
    sma20_1d = close_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient data
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(sma20_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is normal/high
        # Avoid low volatility periods where RSI can give false signals
        if i >= 20:
            atr_ratio = atr_1d_aligned[i] / np.nanmean(atr_1d_aligned[i-20:i])
            vol_filter = not np.isnan(atr_ratio) and atr_ratio > 0.7
        else:
            vol_filter = True
        
        # RSI extreme conditions
        rsi_overbought = rsi_1d_aligned[i] > 70
        rsi_oversold = rsi_1d_aligned[i] < 30
        
        # Price trend filter
        price_above_sma = close_1d[i] > sma20_1d[i]
        price_below_sma = close_1d[i] < sma20_1d[i]
        
        if position == 0:
            # LONG: RSI oversold + price below SMA (mean reversion in downtrend) + volume confirmation
            if rsi_oversold and price_below_sma and vol_filter:
                # Volume confirmation: current volume > 20-period average
                if i >= 20:
                    vol_ma = np.nanmean(volume_1d[i-20:i])
                    vol_conf = volume_1d[i] > vol_ma * 1.2
                else:
                    vol_conf = True
                
                if vol_conf:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            # SHORT: RSI overbought + price above SMA (mean reversion in uptrend) + volume confirmation
            elif rsi_overbought and price_above_sma and vol_filter:
                # Volume confirmation
                if i >= 20:
                    vol_ma = np.nanmean(volume_1d[i-20:i])
                    vol_conf = volume_1d[i] > vol_ma * 1.2
                else:
                    vol_conf = True
                
                if vol_conf:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral or price crosses above SMA
            if rsi_1d_aligned[i] >= 50 or price_above_sma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral or price crosses below SMA
            if rsi_1d_aligned[i] <= 50 or price_below_sma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals