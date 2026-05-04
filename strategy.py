#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width regime filter + 1d RSI mean reversion + volume confirmation
# Long when: BBW(20) < 30th percentile (low volatility squeeze) AND RSI(14) < 30 (oversold) on 1d AND volume > 1.5x 20 EMA
# Short when: BBW(20) < 30th percentile (low volatility squeeze) AND RSI(14) > 70 (overbought) on 1d AND volume > 1.5x 20 EMA
# Uses 6h timeframe for lower frequency, Bollinger Band Width to identify low-volatility regimes ripe for mean reversion,
# 1d RSI for overextension signals, volume confirmation to avoid false breakouts. Designed for 12-37 trades/year with discrete sizing (0.25).
# Works in ranging markets via mean reversion and avoids trending markets via BBW regime filter.

name = "6h_BBW_RSI_Volume_MeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF RSI filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    # Align 1d RSI conditions to 6h timeframe
    rsi_oversold_aligned = align_htf_to_ltf(prices, df_1d, rsi_oversold.astype(float))
    rsi_overbought_aligned = align_htf_to_ltf(prices, df_1d, rsi_overbought.astype(float))
    
    # Calculate 6h Bollinger Band Width (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    bb_width = ((upper_band - lower_band) / sma_20) * 100  # Percentage
    
    # Calculate 20-period volume EMA for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    # Calculate 30th percentile of BBW for regime filter (using expanding window to avoid look-ahead)
    bbw_percentile_30 = np.full_like(bb_width, np.nan)
    for i in range(20, n):  # Start after BBW can be calculated
        if i < 100:  # Need sufficient history for percentile
            bbw_percentile_30[i] = np.percentile(bb_width[20:i+1], 30)
        else:
            bbw_percentile_30[i] = np.percentile(bb_width[i-80:i+1], 30)  # Use last 80 bars for stability
    
    # Low volatility regime: BBW below 30th percentile
    low_volatility_regime = bb_width < bbw_percentile_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(rsi_oversold_aligned[i]) or np.isnan(rsi_overbought_aligned[i]) or 
            np.isnan(low_volatility_regime[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Low volatility regime AND RSI oversold AND volume spike
            if (low_volatility_regime[i] and 
                rsi_oversold_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Low volatility regime AND RSI overbought AND volume spike
            elif (low_volatility_regime[i] and 
                  rsi_overbought_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (50) OR volatility expands OR volume normalizes
            if (rsi_oversold_aligned[i] < 0.5 or  # RSI no longer oversold signal
                not low_volatility_regime[i] or   # Volatility regime changed
                not volume_spike[i]):             # Volume spike ended
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral (50) OR volatility expands OR volume normalizes
            if (rsi_overbought_aligned[i] < 0.5 or  # RSI no longer overbought signal
                not low_volatility_regime[i] or     # Volatility regime changed
                not volume_spike[i]):               # Volume spike ended
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals