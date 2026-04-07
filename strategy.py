#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Keltner Channel Breakout + 12h RSI Filter + Volume Confirmation
# Hypothesis: Keltner Channel (KC) breakouts capture volatility expansion, especially when aligned
# with higher timeframe momentum (12h RSI > 50 for longs, < 50 for shorts) and volume confirmation.
# In strong trends, KC breakouts continue; in ranging markets, false breakouts are filtered by
# 12h RSI and volume. Works in both bull/bear by following momentum on 12h.
# Target: 15-40 trades/year (60-160 over 4 years).
name = "6h_keltner_12h_rsi_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for RSI filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Keltner Channel on 6h (20, 1.5)
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    ema_center = close_s.ewm(span=20, adjust=False, min_periods=20).mean()
    atr = (high_s.rolling(20, min_periods=20).max() - low_s.rolling(20, min_periods=20).min())  # Simple range-based ATR approximation
    atr = atr.ewm(span=20, adjust=False, min_periods=20).mean()  # Smoothed ATR
    upper_keltner = ema_center + (atr * 1.5)
    lower_keltner = ema_center - (atr * 1.5)
    
    # 12-hour RSI(14) for trend filter
    rsi_period = 14
    delta = pd.Series(df_12h['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h = rsi_12h.fillna(50).values  # Neutral when undefined
    rsi_12h_6h = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(rsi_12h_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price re-enters Keltner Channel (mean reversion) or RSI turns bearish
            if close[i] <= upper_keltner[i] and close[i] >= lower_keltner[i] or rsi_12h_6h[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price re-enters Keltner Channel or RSI turns bullish
            if close[i] <= upper_keltner[i] and close[i] >= lower_keltner[i] or rsi_12h_6h[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long: break above upper Keltner with bullish 12h RSI
                if close[i] > upper_keltner[i] and rsi_12h_6h[i] > 50:
                    position = 1
                    signals[i] = 0.25
                # Short: break below lower Keltner with bearish 12h RSI
                elif close[i] < lower_keltner[i] and rsi_12h_6h[i] < 50:
                    position = -1
                    signals[i] = -0.25
    
    return signals