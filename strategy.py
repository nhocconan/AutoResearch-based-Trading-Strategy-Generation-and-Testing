#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1d Keltner Channel breakout with 1w RSI regime filter
# - Uses 1d Keltner Channel (20, 2.0) for dynamic support/resistance levels
# - Uses 1w RSI(14) to identify overbought/oversold conditions for mean reversion in ranging markets
# - Enters long when price breaks below lower KC in oversold weekly RSI (<30) - contrarian bounce
# - Enters short when price breaks above upper KC in overbought weekly RSI (>70) - contrarian fade
# - Uses volume spike confirmation on breakouts to avoid false signals
# - Exits when price returns to KC middle line or RSI returns to neutral zone (40-60)
# - Designed to capture mean reversion moves after extreme weekly RSI readings with intraday breakouts
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "4h_1wRSI_1dKeltner_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner Channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1d Keltner Channel (20, 2.0)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical Price
    tp_1d = (high_1d + low_1d + close_1d) / 3
    
    # ATR(10) for Keltner Channel width
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 10)
    
    # EMA(20) of Typical Price for KC middle line
    ema_tp = pd.Series(tp_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    kc_upper = ema_tp + (2.0 * atr)
    kc_lower = ema_tp - (2.0 * atr)
    kc_middle = ema_tp
    
    # Calculate 1w RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 4h timeframe
    kc_upper_4h = align_htf_to_ltf(prices, df_1d, kc_upper)
    kc_lower_4h = align_htf_to_ltf(prices, df_1d, kc_lower)
    kc_middle_4h = align_htf_to_ltf(prices, df_1d, kc_middle)
    
    # Align 1w RSI to 4h timeframe
    rsi_4h = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Volume filters (4h timeframe)
    vol_ma_10 = pd.Series(volume).ewm(span=10, adjust=False, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(kc_upper_4h[i]) or np.isnan(kc_lower_4h[i]) or np.isnan(kc_middle_4h[i]) or
            np.isnan(rsi_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for extreme weekly RSI and price near Keltner Bands
            oversold = rsi_4h[i] < 30
            overbought = rsi_4h[i] > 70
            
            if oversold:
                # Long: price breaks below lower KC in oversold weekly RSI
                if close[i] < kc_lower_4h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
            elif overbought:
                # Short: price breaks above upper KC in overbought weekly RSI
                if close[i] > kc_upper_4h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to KC middle OR RSI returns to neutral (40-60)
            if close[i] > kc_middle_4h[i] or (rsi_4h[i] >= 40 and rsi_4h[i] <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to KC middle OR RSI returns to neutral (40-60)
            if close[i] < kc_middle_4h[i] or (rsi_4h[i] >= 40 and rsi_4h[i] <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals